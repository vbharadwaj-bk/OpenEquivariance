import json
import cppimport
import cppimport.import_hook
cppimport.settings["use_filelock"] = False
from scipy.sparse import coo_matrix
import numpy as np
import numpy.linalg as la

from src.wrapper.kernel_wrapper import *

if __name__=='__main__':
    rng = np.random.default_rng(12345)

    L1 = 2 # Node feature representations 
    L2 = 2 # Edge feature representations 
    L3 = 2 # Output feature representations

    row = np.array([0, 0, 1, 2, 2, 2], dtype=np.uint64)
    col = np.array([0, 2, 2, 0, 1, 2], dtype=np.uint64)
    data = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint64)
    sparse_mat = coo_matrix((data, (row, col)), shape=(3, 3))

    ctx = ESPMM_Context(sparse_mat.shape[0], L1, L2, L3)

    X_in = np.array(rng.uniform(size=(sparse_mat.shape[0], ctx.get_X_in_rowlen())), dtype=np.float32) 
    edge_features = np.array(rng.uniform(size=(sparse_mat.nnz, ctx.get_edge_rowlen())), dtype=np.float32)
    X_out_cuda_kernel = np.zeros((sparse_mat.shape[1], ctx.get_X_out_rowlen()), dtype=np.float32)

    equivariant_spmm_cpu(ctx,
                    sparse_mat.row,
                    sparse_mat.col,
                    X_in,
                    edge_features,
                    X_out_cuda_kernel)

    ground_truth = sparse_mat @ X_in
    print(la.norm(X_out_cuda_kernel - ground_truth))

