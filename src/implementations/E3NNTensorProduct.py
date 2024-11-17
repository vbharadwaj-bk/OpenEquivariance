import e3nn, torch
import numpy as np
from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *

class E3NNTensorProduct(TensorProduct):
    def __init__(self, config, torch_op=False):
        # Check if config is instance of tensor product problem
        if isinstance(config, TPProblem):
            config = e3nn.o3.TensorProduct(
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

        super().__init__(config, torch_op=torch_op)

    def exec_tensor_product(self,
            batch : np.uint64,
            L1_in: np.uint64,
            L2_in: np.uint64,
            L3_out: np.uint64,
            weights: np.uint64):
        raise NotImplementedError("E3NNTensorProduct does not support exec_tensor_product")

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out, weights):
        L1_in = torch.Tensor(L1_in)
        L2_in = torch.Tensor(L2_in)
        weights = torch.Tensor(weights)
        L3_out[:] = self.config(L1_in, L2_in, weights).detach().numpy()

    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        torch_L1_in = torch.Tensor(L1_in)
        torch_L2_in = torch.Tensor(L2_in)
        torch_weights = torch.Tensor(weights)

        torch_L1_in.requires_grad = True
        torch_L2_in.requires_grad = True
        torch_weights.requires_grad = True

        torch_out = self.config(torch_L1_in, torch_L2_in, torch_weights)
        torch_out.backward(gradient=torch.Tensor(L3_grad))
        
        L1_grad = torch_L1_in.grad.detach().numpy()
        L2_grad = torch_L2_in.grad.detach().numpy()
        weights_grad = torch_weights.grad.detach().numpy()

        return L1_grad, L2_grad, weights_grad

    @staticmethod
    def name():
        return "E3NNTensorProduct"