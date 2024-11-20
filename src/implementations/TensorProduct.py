import pickle, pathlib
from math import prod
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.implementations.e3nn_lite import TPProblem
from src.benchmark.logging_utils import getLogger, bcolors
logger = getLogger()

class GPUInfo:
    A100_SMS = 108
    max_smem = 163840 - 1
    warp_size = 32

def flops_data_per_tp(config, bytes_per_word, direction):
    '''
    Assumes all interactions are "uvu" for now

    Returns (flops_per_tp, data_per_tp, nnz)
    '''
    assert(not config.shared_weights)
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
    ops_per_nz, words_per_tp = None, None
    if direction == "forward":
        ops_per_nz = 3
        words_per_tp = L1.dim + L2.dim + L3.dim + config.weight_numel 
    elif direction == "backward":
        ops_per_nz = 9
        words_per_tp = L1.dim + L2.dim + L3.dim + weights.dim \
                + L1.dim + L2.dim + config.weight_numel # Output gradients

    ops_per_tp = 0
    nnz = 0
    for (u, v, w, *others) in config.instructions:
        tensor = TensorProduct.load_cg_tensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l)
        local_nnz = np.count_nonzero(tensor)
        nnz += local_nnz
        ops_per_tp += ops_per_nz * local_nnz * L1[u].mul * L2[v].mul # Assumes L3.mult(w) = L1.mult(u) * L2.mult(v) 
        ops_per_tp += L3[w].mul * (2 * L3[w].ir.l + 1) # FLOPS for weights, assuming "uvu"

    return ops_per_tp, words_per_tp * bytes_per_word, nnz


class TensorProduct:
    next_tp_id = 0 # Used to assign unique IDs to each TP instance 
    tensors = None
    with open(pathlib.Path("data/CG_tensors.pickle"), 'rb') as f:
        tensors = pickle.load(f) 

    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely.
    '''
    def __init__(self, config : TPProblem, torch_op : bool = False):
        assert isinstance(config, TPProblem)
        assert isinstance(torch_op, bool)
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out

        self.tp_id = TensorProduct.next_tp_id
        TensorProduct.next_tp_id += 1

        if torch_op:
            self.setup_torch_module()

    def forward(
        self,
        batch : np.uint64,
        L1_in: np.uint64,
        L2_in: np.uint64,
        L3_out: np.uint64,
        weights: np.uint64
        ) -> None:
        '''
        Inputs are integers representing device pointers.
        '''
        self.internal.exec_tensor_product(batch, L1_in, L2_in, L3_out, weights) 

    def forward_cpu(
        self, 
        L1_in: np.ndarray, 
        L2_in: np.ndarray, 
        L3_out: np.ndarray, 
        weights: np.ndarray
        ) -> None:
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)

    def backward(self, batch_size: np.uint64,
                L1_in: np.uint64, L1_grad: np.uint64, 
                L2_in: np.uint64, L2_grad: np.uint64,
                weights: np.uint64, weights_grad: np.uint64,
                L3_grad: np.uint64):
        '''
        Inputs are integers representing device pointers.
        '''
        self.internal.backward(
                batch_size,
                L1_in, L1_grad,
                L2_in, L2_grad,
                weights, weights_grad,
                L3_grad)

    def backward_cpu(self, L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad) -> None:
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.backward_cpu(
                L1_in, L1_grad, 
                L2_in, L2_grad,
                weights, weights_grad, 
                L3_grad)

    @staticmethod
    def load_cg_tensor(l1, l2, l3):
        return TensorProduct.tensors[(l1, l2, l3)]

    def benchmark_forward(
        self, 
        num_warmup : int, 
        num_iter : int, 
        L1_in : np.ndarray, 
        L2_in : np.ndarray, 
        L3_buffer : np.ndarray, 
        weights : np.ndarray
        ) -> np.ndarray:
        '''
        Returns the total time for num_iter iterations of the core inner loop forwards
        after num_warmup warmup iterations. Can override for other implementations
        Returns a np array of execution times in milliseconds
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        self.internal.benchmark_forward_cpu(
                    L1_in, L2_in, L3_buffer, weights,
                    num_warmup, time_millis)
            
        return time_millis
    
    def benchmark_backward(
            self, 
            num_warmup : int, 
            num_iter : int, 
            L1_in : np.ndarray, 
            L2_in : np.ndarray, 
            L3_buffer : np.ndarray, 
            weights : np.ndarray, 
            L1_grad : np.ndarray, 
            L2_grad : np.ndarray,
            weights_grad : np.ndarray
            ) -> np.ndarray:
        '''
        Returns the total time for num_iter iterations of the core inner loop backward
        after num_warmup warmup iterations. Can override for other implementations. 
        Returns a np array of execution times in milliseconds
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        self.internal.benchmark_backward_cpu(
                    L1_in, L1_grad,
                    L2_in, L2_grad,
                    weights, weights_grad,
                    L3_buffer,
                    num_warmup, time_millis)
        
        return time_millis

    def calculate_memory_streamed_forward(self, batch_size : int) -> dict: 
        raise NotImplementedError("This needs to be implemented in your class")
    
    def calculate_memory_streamed_backward(self, batch_size : int) -> dict: 
        raise NotImplementedError("This needs to be implemented in your class")
    
    def calculate_flops_forward(self, batch_size : int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")
    
    def calculate_flops_backward(self, batch_size : int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")


    def setup_torch_module(self):
        import torch, typing

        # ----------------- Forward pass -----------------
        @torch.library.custom_op(f"fast_tp::tp_forward{self.tp_id}", mutates_args=(), device_types="cuda")
        def forward(L1_in : torch.Tensor, L2_in : torch.Tensor, weights : torch.Tensor) -> torch.Tensor:
            L1_in_c, L2_in_c, weights_c = L1_in.contiguous(), L2_in.contiguous(), weights.contiguous()
            L3_out = torch.zeros((L1_in_c.shape[0], self.L3.dim ), dtype=torch.float32, device='cuda')
            self.exec_tensor_product(L1_in_c.shape[0], L1_in_c.data_ptr(), L2_in_c.data_ptr(), L3_out.data_ptr(), weights_c.data_ptr())
            return L3_out
        
        @forward.register_fake
        def _(L1_in, L2_in, weights):
            return L1_in.new_empty(L1_in.shape[0], self.L3.dim)
        
        self.forward = forward
        
        # ---------------- Backward pass -----------------
        @torch.library.custom_op(f"fast_tp::tp_grad_helper{self.tp_id}", mutates_args=(), device_types="cuda")
        def grad_helper( L1_in : torch.Tensor, L2_in : torch.Tensor, 
                     weights : torch.Tensor, L3_grad : torch.Tensor ) -> typing.List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.zeros_like(L2_in)
            weights_grad = torch.zeros_like(weights)
            
            self.backward( L1_in.shape[0], L1_in.data_ptr(), L1_grad.data_ptr(),
                        L2_in.data_ptr(), L2_grad.data_ptr(),
                        weights.data_ptr(), weights_grad.data_ptr(),
                        L3_grad.data_ptr() )
            
            return [L1_grad, L2_grad, weights_grad]
        
        @grad_helper.register_fake
        def _(L1_in, L2_in, weights, L3_grad):
            return [L1_in.new_empty(*L1_in.shape), L2_in.new_empty(*L2_in.shape), weights.new_empty(*weights.shape)]

        def setup_context(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights = inputs
        
        def backward(ctx, grad_output):
            result = grad_helper(ctx.L1_in, ctx.L2_in, ctx.weights, grad_output)
            return result[0], result[1], result[2]

        self.forward.register_autograd(backward, setup_context=setup_context)

        # Setup for higher derivatives
        def setup_context_grad_helper(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad = inputs 

        def grad_helper_backward(ctx, grad_output):
            A, B, C, D = ctx.L1_in, ctx.L2_in, ctx.L3_grad, ctx.weights
            E, F, G = grad_output[0], grad_output[1], grad_output[2]

            op1 = grad_helper(A, B, D, C)
            op2 = grad_helper(A, B, G, C)
            op3 = forward(E, B, D)
            op4 = grad_helper(E, B, D, C) # op4 and op5 could be combined with op3 and op6 
            op5 = grad_helper(A, F, D, C) 
            op6 = forward(A, F, D)
            op7 = forward(A, B, G)

            return op1[0] + op2[0], op1[1] + op2[1], op4[2] + op5[2], op3 + op6 + op7

        grad_helper.register_autograd(grad_helper_backward, setup_context=setup_context_grad_helper)