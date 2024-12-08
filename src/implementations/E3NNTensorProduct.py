import e3nn, torch
import numpy as np
from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *

from src.benchmark.logging_utils import getLogger

logger = getLogger()

class E3NNTensorProduct(TensorProduct):
    def __init__(self, config : TPProblem, torch_op=False):
        super().__init__(config, torch_op=torch_op)

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
                    shared_weights=config.shared_weights)

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float32)  # Reset to default

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
        torch_L1_in = torch.tensor(L1_in)
        torch_L2_in = torch.tensor(L2_in)
        torch_weights = torch.tensor(weights)

        torch_L3_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        L3_out[:] = torch_L3_out.detach().numpy()

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

        torch_L1_in = torch.tensor(L1_in).to(device='cuda').detach()
        torch_L2_in = torch.tensor(L2_in).to(device='cuda').detach()

        torch_weights = torch.tensor(weights).to(device='cuda').detach()
        self.e3nn_tp.to(device='cuda')

        for i in range(num_warmup): 
            torch_L3_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        for i in range(num_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch_L3_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)
            end.record()
            torch.cuda.synchronize()
            time_millis[i] = start.elapsed_time(end)
            
        return time_millis

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

        torch_L1_in = torch.tensor(L1_in, requires_grad=True)
        torch_L2_in = torch.tensor(L2_in, requires_grad=True)        
        torch_weights = torch.tensor(weights, requires_grad=True)

        torch_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        torch_L3_grad_in = torch.tensor(L3_grad)

        torch_out.backward(gradient=torch_L3_grad_in)
        
        L1_grad[:] = torch_L1_in.grad.detach().numpy()
        L2_grad[:] = torch_L2_in.grad.detach().numpy()
        weights_grad[:] = torch_weights.grad.detach().numpy()


    def benchmark_backward(self, num_warmup: int, num_iter: int, L1_in: np.ndarray, L2_in: np.ndarray, L3_buffer: np.ndarray, weights: np.ndarray, L1_grad: np.ndarray, L2_grad: np.ndarray, weights_grad: np.ndarray) -> np.ndarray:
        time_millis = np.zeros(num_iter, dtype=np.float32)

        torch_L1_in = torch.tensor(L1_in, requires_grad=True)
        torch_L2_in = torch.tensor(L2_in, requires_grad=True) 
        torch_weights = torch.tensor(weights, requires_grad=True)
        torch_out = self.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)
        torch_L3_grad_in = torch.tensor(L3_buffer)

        for i in range(num_warmup): 
            torch_out.backward(gradient=torch_L3_grad_in, retain_graph=True)

        for i in range(num_iter):
            torch_L1_in.grad.zero_()
            torch_L2_in.grad.zero_()
            torch_weights.grad.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            torch_out.backward(gradient=torch_L3_grad_in, retain_graph=True)

            end.record()
            torch.cuda.synchronize()
            time_millis[i] = start.elapsed_time(end)

        L1_grad[:] = 0.0
        L1_grad[:] = 0.0

        L1_grad[:] = torch_L1_in.grad.detach().numpy()
        L2_grad[:] = torch_L2_in.grad.detach().numpy()
        weights_grad[:] = torch_weights.grad.detach().numpy()

        return time_millis

    @staticmethod
    def name():
        return "E3NNTensorProduct"