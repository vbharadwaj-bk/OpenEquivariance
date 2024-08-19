import json
import cppimport
import cppimport.import_hook
cppimport.settings["use_filelock"] = False
from scipy.sparse import coo_matrix
import numpy as np
import numpy.linalg as la

from src.wrapper.kernel_wrapper import *

if __name__=='__main__':
    L1 = 2 # Node feature representations 
    L2 = 2 # Edge feature representations 
    L3 = 2 # Output feature representations

    rng = np.random.default_rng(12345)
    tp = ThreadTensorProduct(L1, L2, L3)
    batch_size = 100

    L1_in  = np.array(rng.uniform(size=(batch_size, tp.get_row_length(1))), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(batch_size, tp.get_row_length(2))), dtype=np.float32)
    L3_out = np.zeros((batch_size, tp.get_row_length(3)), dtype=np.float32)

    tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out) 
