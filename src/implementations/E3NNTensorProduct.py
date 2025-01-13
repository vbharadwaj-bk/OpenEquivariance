import numpy as np
from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *

from src.benchmark.logging_utils import getLogger

logger = getLogger()

class E3NNTensorProduct(TensorProduct):
    def __init__(self, config : TPProblem, torch_op=True):
        super().__init__(config, torch_op=torch_op)
        assert(self.torch_op)

        global torch
        import torch
        import e3nn 
        assert(config.irrep_dtype == config.weight_dtype)
        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float64)

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
                    shared_weights=config.shared_weights).to(device='cuda')

        logger.info("Torch compiling e3nn TP...")
        self.e3nn_tp = compile(self.e3nn_tp) 
        logger.info("e3nn TP torch compiled.")

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float32)  # Reset to default

        self.forward = self.e3nn_tp.__call__ 

    def forward_cpu(
            self, 
            L1_in : np.ndarray,
            L2_in : np.ndarray, 
            L3_out : np.ndarray, 
            weights : np.ndarray,
            ) -> None:
        torch_L1_in = torch.tensor(L1_in, device='cuda')
        torch_L2_in = torch.tensor(L2_in, device='cuda')
        torch_weights = torch.tensor(weights, device='cuda')
        
        torch_L3_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        L3_out[:] = torch_L3_out.numpy(force=True)

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

        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device='cuda')
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device='cuda')        
        torch_weights = torch.tensor(weights, requires_grad=True, device='cuda')

        torch_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        torch_L3_grad_in = torch.tensor(L3_grad, device='cuda')

        torch_out.backward(gradient=torch_L3_grad_in)
        
        L1_grad[:] = torch_L1_in.grad.numpy(force=True)
        L2_grad[:] = torch_L2_in.grad.numpy(force=True)
        weights_grad[:] = torch_weights.grad.numpy(force=True)

    @staticmethod
    def name():
        return "E3NNTensorProduct"