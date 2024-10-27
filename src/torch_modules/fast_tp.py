import torch
import typing
import e3nn
from e3nn import o3
import torch, typing, sys, os, e3nn

from build.kernel_wrapper import *
from src.implementations.LoopUnrollTP import *

class FastUVUProduct:
    def __init__(self, reps):
        self.internal = LoopUnrollTP(reps, 1)
        
        # ----------------- Forward pass -----------------
        @torch.library.custom_op("fast_tp::forwardUVU", mutates_args=(), device_types="cuda")
        def forward(L1_in : torch.Tensor, L2_in : torch.Tensor, weights : torch.Tensor) -> torch.Tensor:
            L1_in_c, L2_in_c, weights_c = L1_in.contiguous(), L2_in.contiguous(), weights.contiguous()
            L3_out = torch.zeros((L1_in_c.shape[0], self.internal.reps.L3.get_rep_length() ), dtype=torch.float32, device='cuda')
            self.internal.exec_tensor_product(L1_in_c.shape[0], L1_in_c.data_ptr(), L2_in_c.data_ptr(), L3_out.data_ptr(), weights_c.data_ptr())
            return L3_out
        
        @forward.register_fake
        def _(L1_in, L2_in, weights):
            return L1_in.new_empty(L1_in.shape[0], self.internal.reps.L3.get_rep_length())
        
        self.forward = forward
        
        # ---------------- Backward pass -----------------
        @torch.library.custom_op("fast_tp::backwardUVU", mutates_args=(), device_types="cuda")
        def backward_helper( L1_in : torch.Tensor, L2_in : torch.Tensor, 
                     weights : torch.Tensor, L3_grad : torch.Tensor ) -> typing.List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.zeros_like(L2_in)
            weights_grad = torch.zeros_like(weights)
            
            self.internal.backward( L1_in.shape[0], L1_in.data_ptr(), L1_grad.data_ptr(),
                        L2_in.data_ptr(), L2_grad.data_ptr(),
                        weights.data_ptr(), weights_grad.data_ptr(),
                        L3_grad.data_ptr() )
            
            return [L1_grad, L2_grad, weights_grad]
        
        @backward_helper.register_fake
        def _(L1_in, L2_in, weights, L3_grad):
            return [L1_in.new_empty(*L1_in.shape), L2_in.new_empty(*L2_in.shape), weights.new_empty(*weights.shape)]
        
        def setup_context(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights = inputs
        
        def backward(ctx, grad_output):
            result = backward_helper(ctx.L1_in, ctx.L2_in, ctx.weights, grad_output)
            return result[0], result[1], result[2]
        
        self.forward.register_autograd(backward, setup_context=setup_context)


def test_drive():
    torch.set_default_device('cuda')
    irreps_L1 = o3.Irreps("32x5e")
    irreps_L2 = o3.Irreps("1x5e")
    irreps_L3 = o3.Irreps("32x3e")
        
    In1 = irreps_L1.randn(1, -1)
    In2 = irreps_L2.randn(1, -1)
    weights = torch.ones(32, dtype=torch.float32)

    In1.to(device='cuda')
    In2.to(device='cuda')
    weights.to(device='cuda')

    In1.requires_grad_()
    In2.requires_grad_()
    weights.requires_grad_()

    reps = RepTriple(Representation("32x5e"), Representation("1x5e"), Representation("32x3e"))
    fast_tp = FastUVUProduct(reps)
    result_ours = fast_tp.forward(In1, In2, weights)

    random_grad = irreps_L3.randn(1, -1)
    result_ours.backward(random_grad, inputs=[In1, In2])

    print(In1.grad)
    print(In2.grad)