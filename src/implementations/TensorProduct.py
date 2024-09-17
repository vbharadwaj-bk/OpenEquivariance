import pickle, pathlib
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *
from src.benchmark.logging_utils import getLogger, bcolors 

logger = getLogger()

class TensorProduct:
    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely.

    L1, L2, and L3 are pairs. A pair (32, 3) indicates 32 copies of the 
    3-irrep.
    '''
    def __init__(self, L1: Representation, L2: Representation, L3: Representation, batch_size = None):
        self.internal = None
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.batch_size = batch_size

    @staticmethod
    def name():
        raise NotImplementedError()

    def exec_tensor_product(self, batch : int, L1_in, L2_in, L3_out):
        '''
        This function assumes you've already put your arrays on the gpu
        '''
        self.internal.exec_tensor_product(batch, L1_in, L2_in, L3_out) 

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out):
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out) 

    def load_cg_tensor(self, l1, l2, l3):
        with open(pathlib.Path("data/CG_tensors.pickle"), 'rb') as f:
            tensors = pickle.load(f) 
            return tensors[(l1, l2, l3)]

    def test_correctness(self, L1_in, L2_in, L3_out_comp):
        '''
        ATM, this only works for a single multiplicity in each dimension. 
        '''
        L1, L2, L3 = self.L1, self.L2, self.L3

        thresh = 5e-7
        result = {
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": thresh, # Above floating point interval machine epsilon 
            "pass": False
        }

        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        cg_tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        ground_truth = np.einsum('bui,bvj,ijk->buvk', 
                L1_in.reshape((L1_in.shape[0], L1.mult(0), 2 * L1.type(0) + 1)), 
                L2_in.reshape((L2_in.shape[0], L2.mult(0), 2 * L2.type(0) + 1)), 
                cg_tensor)
        ground_truth = ground_truth.reshape(L1_in.shape[0], -1)

        if L3_out_comp.shape != ground_truth.shape:
            result["shape_match"] = False
            logger.error(f"{bcolors.FAIL}Ground truth shape does not match input! {diff_Linf_norm=}, {thresh=} {bcolors.ENDC}")
        else:
            result["shape_match"] = True 
            diff_Linf_norm = float(la.norm((ground_truth - L3_out_comp).flatten(), ord=np.inf))
            result["diff_Linf_norm"] = diff_Linf_norm 
            result["pass"] = bool(diff_Linf_norm < thresh) 

            if result["pass"]:
                logger.info(f"{bcolors.OKGREEN}Correctness check pass. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}Correctness check fail! {diff_Linf_norm=}, {thresh=} {bcolors.ENDC}")

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
        rng = np.random.default_rng(prng_seed)

        L1_in  = np.array(rng.uniform(size=(batch_size, self.L1.get_rep_length())), dtype=np.float32) 
        L2_in  = np.array(rng.uniform(size=(batch_size, self.L2.get_rep_length())), dtype=np.float32)
        L3_out = np.zeros((batch_size, self.L3.get_rep_length()), dtype=np.float32)

        L1, L2, L3 = self.L1, self.L2, self.L3
        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        cg_tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        nnz = len(np.nonzero(cg_tensor)[0])

        # =========== Benchmarking ===========
        time_millis = self.benchmark_internal(num_warmup, num_iter, L1_in, L2_in, L3_out)
        # ==================================== 

        # We don't multiply by num_iters since we benchmark each kernel run separately 
        # Each multiplication requires two multiplications and one addition --> 3 
        ops_per_nz = 3 * self.L1.mult(0)
        throughputs_gflops = [float(el) for el in ops_per_nz * batch_size * nnz / (time_millis * 1e6)]

        bandwidth_gbps_rough = [float(el) for el in (L1_in.nbytes + L2_in.nbytes + L3_out.nbytes) / (time_millis * 1e6)]
        time_millis = [float(el) for el in time_millis] 

        result = {
            "cg tensor nnz": nnz,
            "batch size": batch_size,
            "L1": L1.to_string(),
            "L2": L2.to_string(),
            "L3": L3.to_string(),
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps_rough": bandwidth_gbps_rough
        }
        logger.info(f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")

        return result
