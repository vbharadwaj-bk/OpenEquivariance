import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class ThreadTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3, batch_size):
        super().__init__(L1, L2, L3, batch_size)
        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))

        # Define the sparse tensor in COO format. Coordinate arrays MUST have uint8 datatypes,
        # values must be floats. 
        self.coord= [arr.astype(np.uint8).copy() for arr in np.nonzero(tensor)]
        self.values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        self.internal = ThreadTensorProductImpl(L1, L2, L3, self.coord[0], self.coord[1], self.coord[2], self.values)


    @staticmethod
    def name():
        return "ThreadTensorProduct"