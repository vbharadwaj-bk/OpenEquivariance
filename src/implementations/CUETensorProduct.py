import numpy as np
from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *
from src.benchmark.logging_utils import getLogger

logger = getLogger()

class CUETensorProduct(TensorProduct):
    def __init__(self, config : TPProblem, torch_op=False):
        super().__init__(config, torch_op=torch_op)

        import torch
        import cuequivariance as cue
        import cuequivariance_torch as cuet

        # Currently, we only support channelwise tensor products.
        # Can expand to include self-connection layers 
        assert(isinstance(config, ChannelwiseTensorProduct))

        e = cue.descriptors.channelwise_tensor_product(
            cue.Irreps("O3", str(config.irreps_in1)),
            cue.Irreps("O3", str(config.irreps_in2)) 
            cue.Irreps("O3", str(config.irreps_out))
        )

        self.cue_tp = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul)
        self.cue_tp.to('cuda')

        assert(config.weight_numel == e.inputs[0].irreps.dim)
        
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
        after num_warmup warmup iterations. Can override for other implementations
        Returns a np array of execution times in milliseconds
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        torch_L1_in = torch.Tensor(L1_in).to(device='cuda').detach()
        torch_L2_in = torch.Tensor(L2_in).to(device='cuda').detach()
        torch_weights = torch.Tensor(weights).to(device='cuda').detach()

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


    def benchmark_backward(self, num_warmup: int, num_iter: int, L1_in: np.ndarray, L2_in: np.ndarray, L3_buffer: np.ndarray, weights: np.ndarray, L1_grad: np.ndarray, L2_grad: np.ndarray, weights_grad: np.ndarray) -> np.ndarray:
        return time_millis

    @staticmethod
    def name():
        return "E3NNTensorProduct"