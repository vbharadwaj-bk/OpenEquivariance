import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class ThreadTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3):
        super().__init__(L1, L2, L3)
        self.internal = ThreadTensorProductImpl(L1, L2, L3)

        tensor = e3nn.o3.wigner_3j(l1, l2, l3).numpy()
