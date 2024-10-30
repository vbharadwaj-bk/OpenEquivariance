import pickle, pathlib
from math import prod
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *
from built.kernel_wrapper import Representation, RepTriple

from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

import e3nn

class GPUInfo:
    A100_SMS = 108
    max_smem = 163840 - 1
    warp_size = 32

class TensorProduct:
    tensors = None
    with open(pathlib.Path("data/CG_tensors.pickle"), 'rb') as f:
        tensors = pickle.load(f) 

    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely.
    '''
    def __init__(self, e3nn_tp : e3nn.o3.TensorProduct):
        self.e3nn_tp = e3nn_tp
        self.L1 = Representation(str(e3nn_tp.irreps_in1))
        self.L2 = Representation(str(e3nn_tp.irreps_in2))
        self.L3 = Representation(str(e3nn_tp.irreps_out))
        # self.reps was replaced with a property indicating that it should not be used
        
        # self.batch_size = batch_size

    @property
    def reps():
        raise NotImplementedError("This property has been depricated, please don't use")

    @property
    def batch_size():
        raise NotImplementedError("This property has been depricated, please don't use")
    
    @staticmethod
    def name():
        raise NotImplementedError()

    def forward_cpu(self, L1_in, L2_in, L3_out, weights) -> None:
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights) 

    def backward_cpu(self, L1_in, L1_grad, L2_in, L2_grad, L3_grad, weights, weights_grad) -> None:
        '''
        We will return to the convention of having all bufferrs supplied by the user
        '''
        self.internal.backward_cpu(
                L1_in, L1_grad, 
                L2_in, L2_grad,
                weights, weights_grad, 
                L3_grad)

        return L1_grad, L2_grad, weights_grad

    def load_cg_tensor(self, l1 : int, l2 : int, l3 : int) -> np.ndarray:
        return TensorProduct.tensors[(l1, l2, l3)]

    def benchmark_internal_forward(self, num_warmup, num_iter, L1_in, L2_in, weights, L3_buffer) -> np.ndarray:
        '''
        Returns the total time for num_iter iterations of the core inner loop forwards
        after num_warmup warmup iterations. Can override for other implementations
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        self.internal.benchmark_forward_cpu(
            L1_in, 
            L2_in,
            weights, 
            L3_buffer,    
            num_warmup, 
            time_millis
        )

        return time_millis
    
    def benchmark_internal_backward(self, num_warmup, num_iter, L1_in, L2_in, weights, L3_grad, L1_grad, L2_grad, weights_grad) -> np.ndarray:
        '''
        Returns the total time for num_iter iterations of the core inner loop backwards
        after num_warmup warmup iterations. Can override for other implementations
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        self.internal.benchmark_backward_cpu(
            L1_in, 
            L1_grad,
            L2_in, 
            L2_grad,
            weights, 
            weights_grad,
            L3_grad,
            num_warmup, 
            time_millis
            )

        return time_millis

    @staticmethod
    def calculate_data_streamed_forward(self, batch_size : int) -> dict: 
        raise NotImplementedError("This needs to be implemented in your class")
    
    @staticmethod
    def calculate_data_streamed_backward(self, batch_size : int) -> dict: 
        raise NotImplementedError("This needs to be implemented in your class")
    
    @staticmethod
    def calculate_flops_forward(self, batch_size : int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")
    
    @staticmethod
    def calculate_flops_backward(self, batch_size : int) -> dict:
        raise NotImplementedError("This needs to be implemented in your class")

    