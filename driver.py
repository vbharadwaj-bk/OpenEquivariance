import json
import cppimport
import cppimport.import_hook
cppimport.settings["use_filelock"] = False
from scipy.sparse import coo_matrix
import numpy as np
import numpy.linalg as la

from src.wrapper.kernel_wrapper import *
from src.implementations.ThreadTensorProduct import *

if __name__=='__main__':
    L1 = 5 # Node feature representations 
    L2 = 5 # Edge feature representations 
    L3 = 3 # Output feature representations

    rng = np.random.default_rng(12345)
    tp = ThreadTensorProduct(L1, L2, L3)

    batch_size = 1000000
    L1_in  = np.array(rng.uniform(size=(batch_size, tp.get_row_length(1))), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(batch_size, tp.get_row_length(2))), dtype=np.float32)
    L3_out = np.zeros((batch_size, tp.get_row_length(3)), dtype=np.float32)

    print("Starting tensor product execution!")
    tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
    print("Finished GPU Execution")
    result, ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)

    benchmark = tp.benchmark(10, 30, 10000000, prng_seed=12345) 
    
    print(result)
    print(benchmark)
