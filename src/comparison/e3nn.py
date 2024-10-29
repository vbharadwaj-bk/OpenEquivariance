import torch, e3nn
from e3nn import o3

import sys, os
from build.kernel_wrapper import *
from src.implementations.LoopUnrollTP import *
from src.torch_modules.irrep_transposer import *

def compare_output_to_e3nn(config, batch_size):
    torch.set_default_device('cuda')
        
    irreps_L1, irreps_L2, irreps_L3 = o3.Irreps("32x5e"), o3.Irreps("1x5e"), o3.Irreps("32x3e")
        
    batch_size = 1
    In1 = irreps_L1.randn(batch_size, -1)
    In2 = irreps_L2.randn(batch_size, -1)
    weights = torch.ones(batch_size, reps.num_trainable_weights(), dtype=torch.float32) * 3

    In1.to(device='cuda')
    In2.to(device='cuda')
    weights.to(device='cuda')

    In1.requires_grad_()
    In2.requires_grad_()
    weights.requires_grad_()

    reps = RepTriple(Representation("32x5e"), Representation("1x5e"), Representation("32x3e"))
    L1T = IrrepTransposer(reps.L1)
    L3T = IrrepTransposer(reps.L3)

    fast_tp = LoopUnrollTP(reps, torch_op=True)
    result_ours = L3T(fast_tp.forward(L1T(In1, True), In2, weights), False)

    instructions = [reps.interactions(i) for i in range(reps.num_interactions())]
    instructions = [(i[0], i[1], i[2], "uvu", True) for i in instructions]
    tp = o3.TensorProduct(irreps_L1, irreps_L2, irreps_L3, instructions=instructions, internal_weights=False, shared_weights=False) # internal_weights=False, shared_weights=False
    result_e3nn = tp(In1, In2, weights)

    print(result_ours / result_e3nn)

    random_grad = irreps_L3.randn(1, -1)
    result_ours.backward(random_grad, inputs=[In1, In2, weights])
    grad1_ours = In1.grad.clone().detach()
    grad2_ours = In2.grad.clone().detach()
    gradweight_ours = weights.grad.clone().detach()
    In1.grad.zero_()
    In2.grad.zero_()
    weights.grad.zero_()

    result_e3nn.backward(random_grad, inputs=[In1, In2, weights])
    grad1_e3nn = In1.grad.clone().detach()
    grad2_e3nn = In2.grad.clone().detach()
    gradweight_e3nn = weights.grad.clone().detach()

    print(grad1_e3nn / grad1_ours)
    print(grad2_e3nn / grad2_ours)
    print(gradweight_e3nn / gradweight_ours)

def test_drive_torch_module():
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

    L1T = IrrepTransposer(reps.L1)
    L3T = IrrepTransposer(reps.L3)

    reps = RepTriple(Representation("32x5e"), Representation("1x5e"), Representation("32x3e"))
    fast_tp = LoopUnrollTP(reps, torch_op=True)
    result_ours = L3T(fast_tp.forward(L1T(In1, True), In2, weights), False)

    random_grad = irreps_L3.randn(1, -1)
    result_ours.backward(random_grad, inputs=[In1, In2])
