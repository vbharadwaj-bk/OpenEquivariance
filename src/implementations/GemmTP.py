import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class GemmTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3):
        super().__init__(L1, L2, L3)
        tensor = self.cg_tensor 
        self.flat_tensor = None
        self.internal = GemmTensorProductImpl(L1, L2, L3, self.flat_tensor)


    @staticmethod
    def name():
        return "ThreadTensorProduct"