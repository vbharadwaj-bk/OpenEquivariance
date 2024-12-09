import numpy as np

from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *
from src.benchmark.logging_utils import getLogger
from src.benchmark.tpp_creation_utils import *

logger = getLogger()

class CUETensorProduct(TensorProduct):
    def __init__(self, config : TPProblem):
        super().__init__(config, torch_op=True)

        global torch
        import torch
        import cuequivariance as cue
        import cuequivariance_torch as cuet

        supported_tpp_types = [
            ChannelwiseTPP,
            FullyConnectedTPProblem,
            SingleInstruction
        ]

        assert(config.irrep_dtype == config.weight_dtype)

        np_to_torch_dtype = {
            np.float32: torch.float32,
            np.float64: torch.float64
        }

        assert(any([isinstance(config, supported_ttp_type)] for supported_ttp_type in supported_tpp_types))
        if isinstance(config, ChannelwiseTPP) or isinstance(config, SingleInstruction):
            e = cue.descriptors.channelwise_tensor_product(
                cue.Irreps("O3", str(config.irreps_in1)),
                cue.Irreps("O3", str(config.irreps_in2)),
                cue.Irreps("O3", str(config.irreps_out)))
        
        if isinstance(config, FullyConnectedTPProblem):
            e = cue.descriptors.fully_connected_tensor_product(
                cue.Irreps("O3", str(config.irreps_in1)),
                cue.Irreps("O3", str(config.irreps_in2)),
                cue.Irreps("O3", str(config.irreps_out)),
            )

        assert(config.weight_numel == e.inputs[0].irreps.dim)
        self.cue_tp = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul, math_dtype=np_to_torch_dtype[config.irrep_dtype])        
        self.cue_tp.to('cuda')
        
    def forward(self,
            batch : np.uint64,
            L1_in: np.uint64,
            L2_in: np.uint64,
            L3_out: np.uint64,
            weights: np.uint64,
            ) -> None:
        raise NotImplementedError("CUETensorProduct does not support forward")

    def forward_cpu(
            self, 
            L1_in : np.ndarray,
            L2_in : np.ndarray, 
            L3_out : np.ndarray, 
            weights : np.ndarray,
            ) -> None:
        raise NotImplementedError("CUETensorProduct does not support forward CPU")

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
        after num_warmup warmup iterations.
        Returns a np array of execution times in milliseconds
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        torch_L1_in = torch.tensor(L1_in, device='cuda')
        torch_L2_in = torch.tensor(L2_in, device='cuda')
        torch_weights = torch.tensor(weights, device='cuda')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in range(num_warmup): 
            torch_L3_out = self.cue_tp(torch_weights, torch_L1_in, torch_L2_in, use_fallback=False) 

        for i in range(num_iter):
            start.record()
            torch_L3_out = self.cue_tp(torch_weights, torch_L1_in, torch_L2_in, use_fallback=False) 
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
        raise NotImplementedError("CUETensorProduct does not support backward")

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
        raise NotImplementedError("CUETensorProduct does not support backward_cpu!")


    def benchmark_backward(
            self, 
            num_warmup: int, 
            num_iter: int, 
            L1_in: np.ndarray, 
            L2_in: np.ndarray, 
            L3_buffer: np.ndarray, 
            weights: np.ndarray, 
            L1_grad: np.ndarray, 
            L2_grad: np.ndarray, 
            weights_grad: np.ndarray
            ) -> np.ndarray:
        
        time_millis = np.zeros(num_iter, dtype=np.float32)

        torch_L1_in = torch.tensor(L1_in, requires_grad=True, device='cuda')
        torch_L2_in = torch.tensor(L2_in, requires_grad=True, device='cuda') 
        torch_weights = torch.tensor(weights, requires_grad=True, device='cuda')

        torch_out = self.cue_tp(torch_weights, torch_L1_in, torch_L2_in, use_fallback=False)

        torch_L3_grad_in = torch.tensor(L3_buffer, device='cuda')

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

        L1_grad[:] = torch_L1_in.grad.numpy(force=True)
        L2_grad[:] = torch_L2_in.grad.numpy(force=True)
        weights_grad[:] = torch_weights.grad.numpy(force=True)

        return time_millis
        
    @staticmethod
    def name():
        return "CUETensorProduct"