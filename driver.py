import json
import cppimport
import cppimport.import_hook
from scipy.sparse import csr_matrix

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
    X_in = rng.uniform(size=(sparse_mat.shape[0], 2 * L1 + 1)) 
    edge_features = np.array(rng.uniform(size=(sparse_mat.nnz, 2 * L2 + 1)), dtype=np.double)
    X_out_cuda_kernel = np.zeros((sparse_mat.shape[1], 2 * L2 + 1), dtype=np.double)

    equivariant_spmm(L1, L2, L3,
                    sparse_mat.indptr,
                    sparse_mat.indices,
                    X_in,
                    edge_features,
                    X_out_cuda_kernel)



