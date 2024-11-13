import e3nn, torch
import numpy as np
from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *

from src.benchmark.logging_utils import getLogger

logger = getLogger()

class E3NNTensorProduct(TensorProduct):
    def __init__(self, config : TPProblem, torch_op=False):
        super().__init__(config, torch_op=torch_op)
        self.e3nn_tp = e3nn.o3.TensorProduct(
                    config.irreps_in1, 
                    config.irreps_in2, 
                    config.irreps_out, 
                    config.instructions_raw,
                    in1_var=config.in1_var,
                    in2_var=config.in2_var,
                    out_var=config.out_var,
                    irrep_normalization=config.irrep_normalization,
                    path_normalization=config.path_normalization,
                    internal_weights=config.internal_weights,
                    shared_weights=config.shared_weights)
        
    def forward(self,
            batch : np.uint64,
            L1_in: np.uint64,
            L2_in: np.uint64,
            L3_out: np.uint64,
            weights: np.uint64,
            ) -> None:
        raise NotImplementedError("E3NNTensorProduct does not support forward")

    def forward_cpu(
            self, 
            L1_in : np.ndarray,
            L2_in : np.ndarray, 
            L3_out : np.ndarray, 
            weights : np.ndarray,
            ) -> None:
        torch_L1_in = torch.Tensor(L1_in)
        torch_L2_in = torch.Tensor(L2_in)
        torch_weights = torch.Tensor(weights)

        torch_L3_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)
        
        L3_out[:] = torch_L3_out.detach().numpy().flatten()

    def benchmark_forward(
            self, 
            num_warmup: int, 
            num_iter: int, 
            L1_in: np.ndarray, 
            L2_in: np.ndarray, 
            L3_buffer: np.ndarray, 
            weights: np.ndarray
            ) -> np.ndarray:
        raise NotImplementedError("E3NNTensorProduct does not support benchmark forward")
    
    def backward(self, batch_size: np.uint64,
                L1_in: np.uint64, L1_grad: np.uint64, 
                L2_in: np.uint64, L2_grad: np.uint64,
                weights: np.uint64, weights_grad: np.uint64,
                L3_grad: np.uint64
                ) -> None:
        raise NotImplementedError("E3NNTensorProduct does not support backward")

    def backward_cpu(
            self,
            L1_in : np.ndarray,
            L1_grad : np.ndarray,
            L2_in : np.ndarray,
            L2_grad : np.ndarray,
            L3_grad : np.ndarray,
            weights : np.ndarray,
            weights_grad : np.ndarray,
            ) -> None:
        torch_L1_in = torch.Tensor(L1_in, requires_grad=True)
        torch_L2_in = torch.Tensor(L2_in, requires_grad=True)
        
        torch_weights = torch.Tensor(weights, requires_grad=True)

        torch_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        torch_L3_grad_in = torch.Tensor(L3_grad)

        torch_out.backward(gradient=torch_L3_grad_in)
        
        L1_grad[:] = torch_L1_in.grad.detach().numpy().flatten()
        L2_grad[:] = torch_L2_in.grad.detach().numpy().flatten()
        weights_grad[:] = torch_weights.grad.detach().numpy().flatten()

    def benchmark_backward(self, num_warmup: int, num_iter: int, L1_in: np.ndarray, L2_in: np.ndarray, L3_buffer: np.ndarray, weights: np.ndarray, L1_grad: np.ndarray, L2_grad: np.ndarray, weights_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("E3NNTensorProduct does not support benchmark backward")