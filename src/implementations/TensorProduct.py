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

    def benchmark_internal(self, num_warmup, num_iter, batch_size):
        '''
        Implement to return the total time for num_iter iterations of the core inner loop
        after num_warmup warmup iterations. 
        '''
        raise NotImplementedError()

    def benchmark(self, num_warmup, num_iter, batch_size, prng_seed):
        '''
        This function only works for scalar L-values right now, need to change
        to handle any multiplicity.
        '''
        assert(isinstance(self.L1, int)) 

        nnz = len(np.nonzero(self.cg_tensor)[0])
        times = self.benchmark_internal(num_warmup, num_iter, batch_size)
        throughputs = batch_size * num_iter * nnz / times        

        result = {
            "cg tensor nnz": nnz,
            "batch size": batch_size,
            "L1": self.L1,
            "L2": self.L2,
            "L3": self.L3,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "times": times,
            "throughputs": throughputs 
        }

        return result

