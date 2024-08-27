import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class ThreadTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3):
        super().__init__(L1, L2, L3)
        self.name = "ThreadTensorProduct"

        # Define the sparse tensor in COO format. Coordinate arrays MUST have uint8 datatypes,
        # values must be floats. 

        tensor = self.cg_tensor
        self.coord= [arr.astype(np.uint8).copy() for arr in np.nonzero(tensor)]
        self.values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        self.internal = ThreadTensorProductImpl(L1, L2, L3, self.coord[0], self.coord[1], self.coord[2], self.values)