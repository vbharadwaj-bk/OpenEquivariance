import pickle
import numpy as np
import numpy.linalg as la

class TensorProduct:
    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely. 
    '''
    def __init__(self, L1, L2, L3):
        self.internal = None
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        self.cg_tensor = self.load_cg_tensor(self.L1, self.L2, self.L3)

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out):
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out) 

    def get_row_length(self, mode):
        return self.internal.get_row_length(mode)

    def load_cg_tensor(self, l1, l2, l3):
        with open("data/CG_tensors.pickle", 'rb') as f:
            tensors = pickle.load(f) 
            return tensors[(l1, l2, l3)]

    def test_correctness(self, L1_in, L2_in, L3_out_comp):
        '''
        ATM, this only works for a single multiplicity in each dimension. 
        '''
        result = {
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": 5e-7, # Above floating point interval machine epsilon 
            "pass": False
        }

        ground_truth = np.einsum('bi,bj,ijk->bk', L1_in, L2_in, self.cg_tensor)

        if L3_out_comp.shape != ground_truth.shape:
            result["shape_match"] = False
        else:
            result["shape_match"] = True 
            diff_norm = la.norm((ground_truth - L3_out_comp).flatten(), ord=np.inf)
            result["diff_Linf_norm"] = diff_norm
            result["pass"] = diff_norm < result["thresh"]

        return result, ground_truth

    def benchmark_internal(self, num_warmup, num_iter, L1_in, L2_in, L3_out):
        '''
        Returns the total time for num_iter iterations of the core inner loop
        after num_warmup warmup iterations. Can override for other implementations
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)
        self.internal.benchmark_cpu(
                L1_in,
                L2_in,
                L3_out,
                num_warmup,
                time_millis)

        return time_millis


    def benchmark(self, num_warmup, num_iter, batch_size, prng_seed=12345):
        '''
        This function only works for scalar L-values right now, need to change
        to handle any multiplicity.
        '''
        assert(isinstance(self.L1, int)) 

        rng = np.random.default_rng(prng_seed)

        L1_in  = np.array(rng.uniform(size=(batch_size, self.get_row_length(1))), dtype=np.float32) 
        L2_in  = np.array(rng.uniform(size=(batch_size, self.get_row_length(2))), dtype=np.float32)
        L3_out = np.zeros((batch_size, self.get_row_length(3)), dtype=np.float32)

        nnz = len(np.nonzero(self.cg_tensor)[0])
        time_millis = self.benchmark_internal(num_warmup, num_iter, L1_in, L2_in, L3_out)

        # We don't multiply by num_iters since we benchmark each kernel run separately 
        # Each multiplication requires two multiplications and one addition --> 3 
        ops_per_nz = 3
        throughputs_gflops = ops_per_nz * batch_size * nnz / (time_millis * 1e6)
        bandwidth_gbps_rough = (L1_in.nbytes + L2_in.nbytes + L3_out.nbytes) / (time_millis * 1e6)

        result = {
            "cg tensor nnz": nnz,
            "batch size": batch_size,
            "L1": self.L1,
            "L2": self.L2,
            "L3": self.L3,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps_rough": bandwidth_gbps_rough
        }

        return result
