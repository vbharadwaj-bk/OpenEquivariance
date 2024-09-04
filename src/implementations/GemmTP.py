import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class GemmTensorProduct(TensorProduct):
    def __init__(self, batch_size, L1, L2, L3):
        super().__init__(batch_size, L1, L2, L3)

        if L1[0] != 1 or L2[0] != 1 or L3[0] != 1:
            raise NotImplementedError()
    
        tensor = self.cg_tensor 
        self.flat_tensor = tensor.reshape(((2 * L1[1] + 1) * (2 * L2[1] + 1), 2 * L3[1] + 1)).T.copy() 
        self.internal = GemmTensorProductImpl(batch_size, L1[1], L2[1], L3[1], self.flat_tensor)


    @staticmethod
    def name():
        return "GemmTensorProduct"