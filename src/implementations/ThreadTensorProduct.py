import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class ThreadTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3):
        super().__init__(L1, L2, L3)

        # Define the sparse tensor in COO format. Coordinate arrays MUST have uint8 datatypes,
        # values must be floats.
        self.coord1 = np.arange(2 * L1 + 1, dtype=np.uint8)
        self.coord2 = np.arange(2 * L1 + 1, dtype=np.uint8) 
        self.coord3 = np.arange(2 * L1 + 1, dtype=np.uint8)
        self.values = np.ones(2 * L1 + 1, dtype=np.float32)

        self.internal = ThreadTensorProductImpl(L1, L2, L3, self.coord1, self.coord2, self.coord3, self.values)
        #tensor = e3nn.o3.wigner_3j(l1, l2, l3).numpy()




