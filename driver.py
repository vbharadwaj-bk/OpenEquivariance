import json
import cppimport
import cppimport.import_hook
from scipy.sparse import csr_matrix
import numpy as np

from src.wrapper.kernel_wrapper import *

if __name__=='__main__':
    rng = np.random.default_rng(12345)

    L1 = 3 # Node feature representations 
    L2 = 3 # Edge feature representations 
    L3 = 3 # Output feature representations

    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sparse_mat = csr_matrix((data, (row, col)), shape=(3, 3))

    ctx = ESPMM_Context(sparse_mat.shape[0], L1, L2, L3)

    X_in = np.array(rng.uniform(size=(sparse_mat.shape[0], ctx.get_X_in_rowlen())), dtype=np.float) 
    edge_features = np.array(rng.uniform(size=(sparse_mat.nnz, ctx.get_edge_rowlen())), dtype=np.float)
    X_out_cuda_kernel = np.zeros((sparse_mat.shape[1], ctx.get_X_out_rowlen()), dtype=np.float)

    equivariant_spmm_cpu(ctx,
                    sparse_mat.indptr,
                    sparse_mat.indices,
                    X_in,
                    edge_features,
                    X_out_cuda_kernel)






