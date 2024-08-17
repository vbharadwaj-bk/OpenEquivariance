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
    ctx = TensorProduct(L1, L2, L3)
    batch_size = 100

    X_in = np.array(rng.uniform(size=(batch_size, ctx.get_X_in_rowlen())), dtype=np.float32) 
    edge_features = np.array(rng.uniform(size=(batch_size, ctx.get_edge_rowlen())), dtype=np.float32)
    X_out_cuda_kernel = np.zeros((batch_size, ctx.get_X_out_rowlen()), dtype=np.float32)
