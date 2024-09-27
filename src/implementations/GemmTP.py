import numpy as np
from build.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class GemmTensorProduct(TensorProduct):
    def __init__(self, reps, batch_size):
        super().__init__(reps, batch_size)
        L1, L2, L3 = self.L1, self.L2, self.L3
        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        assert(L1.mult(0) == 1 and L2.mult(0) == 1 and L3.mult(0) == 1)

        tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        self.flat_tensor = tensor.reshape(((2 * L1.type(0) + 1) * (2 * L2.type(0) + 1), 2 * L3.type(0) + 1)).T.copy() 
        self.internal = GemmTensorProductImpl(batch_size, self.reps, self.flat_tensor)

    @staticmethod
    def name():
        return "GemmTensorProduct"