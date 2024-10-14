import pickle, pathlib
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

class GPUInfo:
    A100_SMS = 108
    max_smem = 163840 - 1
    warp_size = 32

class TensorProduct:
    tensors = None
    with open(pathlib.Path("data/CG_tensors.pickle"), 'rb') as f:
        tensors = pickle.load(f) 

    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely.
    '''
    def __init__(self, reps: RepTriple, batch_size=None):
        self.reps = reps 
        self.L1, self.L2, self.L3 = reps.L1, reps.L2, reps.L3
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

    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        '''
        We break from convention here by allocating and returning
        the appropriate buffers. 
        '''
        L1_grad = np.zeros_like(L1_in)
        L2_grad = np.zeros_like(L2_in)
        weights_grad = np.zeros_like(weights)

        self.internal.backward_cpu(L1_in, L1_grad, 
                L2_in, L2_grad,
                weights, weights_grad, 
                L3_grad)

        return L1_grad, L2_grad, weights_grad

    def load_cg_tensor(self, l1, l2, l3):
        return TensorProduct.tensors[(l1, l2, l3)]

    def test_correctness(self, L1_in, L2_in, L3_out_comp):
        thresh = 5e-7
        result = {
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": thresh, # Above floating point interval machine epsilon 
            "pass": False
        }

        L1, L2, L3 = self.L1, self.L2, self.L3
        reps = self.reps
        offsets = { 1: L1.get_irrep_offsets(), 
                    2: L2.get_irrep_offsets(), 
                    3: L3.get_irrep_offsets() }

        ground_truth = np.zeros((L1_in.shape[0], L3.get_rep_length()), dtype=np.float32)

        for i in range(reps.num_interactions()):
            irr1, irr2, irr3 = reps.interactions(i)
            cg_tensor = self.load_cg_tensor(L1.type(irr1), L2.type(irr2), L3.type(irr3))
            start1, end1 = offsets[1][irr1], offsets[1][irr1+1]
            start2, end2 = offsets[2][irr2], offsets[2][irr2+1]
            start3, end3 = offsets[3][irr3], offsets[3][irr3+1]

            ground_truth[:, start3:end3] += np.einsum('bui,bvj,ijk->buvk', 
                    L1_in[:, start1:end1].reshape((L1_in.shape[0], L1.mult(irr1), 2 * L1.type(irr1) + 1)),
                    L2_in[:, start2:end2].reshape((L2_in.shape[0], L2.mult(irr2), 2 * L2.type(irr2) + 1)),
                    cg_tensor).reshape(L1_in.shape[0], -1)

        if L3_out_comp.shape != ground_truth.shape:
            result["shape_match"] = False
            logger.error(f"{bcolors.FAIL}Ground truth shape does not match input! {L3_out_comp.shape=}, {ground_truth.shape=} {bcolors.ENDC}")
        else:
            result["shape_match"] = True 
            diff_Linf_norm = float(la.norm((ground_truth - L3_out_comp).flatten(), ord=np.inf))
            result["diff_Linf_norm"] = diff_Linf_norm 
            result["pass"] = bool(diff_Linf_norm < thresh) 

            if result["pass"]:
                logger.info(f"{bcolors.OKGREEN}Batch TP correctness check pass. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}Batch TP correctness check fail! {diff_Linf_norm=}, {thresh=} {bcolors.ENDC}")

        return result, ground_truth

    def benchmark_internal(self, num_warmup, num_iter, L1_in, L2_in, L3_buffer, weights, L1_grad, L2_grad, weights_grad, direction):
        '''
        Returns the total time for num_iter iterations of the core inner loop
        after num_warmup warmup iterations. Can override for other implementations
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        if direction == "forward":
            self.internal.benchmark_forward_cpu(
                    L1_in, L2_in, L3_buffer,
                    num_warmup, time_millis)
        
        elif direction == "backward":
            self.internal.benchmark_backward_cpu(
                    L1_in, L1_grad,
                    L2_in, L2_grad,
                    weights, weights_grad,
                    L3_buffer,
                    num_warmup, time_millis)

        return time_millis

    def benchmark(self, num_warmup, num_iter, batch_size, direction, prng_seed=12345):
        '''
        This function only works for scalar L-values right now, need to change
        to handle any multiplicity.
        '''
        assert(direction == "forward" or direction == "backward")
        rng = np.random.default_rng(prng_seed)
        L1, L2, L3 = self.L1, self.L2, self.L3
        interactions = [self.reps.interactions(i) for i in range(self.reps.num_interactions())] 

        L1_in  = np.array(rng.uniform(size=(batch_size, self.L1.get_rep_length())), dtype=np.float32) 
        L2_in  = np.array(rng.uniform(size=(batch_size, self.L2.get_rep_length())), dtype=np.float32)
        L3_buffer = np.zeros((batch_size, self.L3.get_rep_length()), dtype=np.float32)

        weights, L1_grad, L2_grad, weights_grad = [None] * 4
        if direction == "backward":
            L3_buffer[:] = rng.uniform(size=(batch_size, L3.get_rep_length())) 
            weights = np.array(rng.uniform(size=(batch_size, self.reps.num_trainable_weights())), dtype=np.float32)

            L1_grad = np.zeros_like(L1_in)
            L2_grad = np.zeros_like(L2_in)
            weights_grad = np.zeros_like(weights)

        logger.info("Initialized input / output data.")

        # Forward: Requires two multiplications and one addition --> 3, 4 if weights are included (not yet)
        # Backward: Requires 6 multiplications and 3 additions (including the weight, implemented)
        ops_per_nz, total_data_streamed  = None, None
        if direction == "forward":
            ops_per_nz = 3
            total_data_streamed = L1_in.nbytes + L2_in.nbytes + L3_buffer.nbytes 
        elif direction == "backward":
            ops_per_nz = 9
            total_data_streamed = L1_in.nbytes + L2_in.nbytes + L3_buffer.nbytes + weights.nbytes \
                    + L1_grad.nbytes + L2_grad.nbytes + weights_grad.nbytes

        ops_per_tp = 0
        nnz = 0
        for u, v, w in interactions:
            tensor = self.load_cg_tensor(L1.type(u), L2.type(v), L3.type(w))
            local_nnz = np.count_nonzero(tensor)
            nnz += local_nnz
            ops_per_tp += ops_per_nz * local_nnz * L1.mult(u) * L2.mult(v) # Assumes L3.mult(w) = L1.mult(u) * L2.mult(v) 

        # =========== Benchmarking ===========
        time_millis = self.benchmark_internal(num_warmup, num_iter, L1_in, L2_in, L3_buffer, weights, L1_grad, L2_grad, weights_grad, direction)
        # ==================================== 

        # We don't multiply by num_iters since we benchmark each kernel run separately 
        # Each multiplication requires two multiplications and one addition --> 3 
        ops_per_nz = 3
        throughputs_gflops = [float(el) for el in ops_per_nz * batch_size * nnz / (time_millis * 1e6)]

        bandwidth_gbps_rough = [float(el) for el in (L1_in.nbytes + L2_in.nbytes + L3_out.nbytes) / (time_millis * 1e6)]
        time_millis = [float(el) for el in time_millis] 

        result = {
            "tp_direction": direction,
            "total_cg_nnz": nnz,
            "flops_per_tp": ops_per_tp,
            "L1": L1.to_string(),
            "L2": L2.to_string(),
            "L3": L3.to_string(),

            "L1_rep_len": L1.get_rep_length(),
            "L2_rep_len": L2.get_rep_length(),
            "L3_rep_len": L3.get_rep_length(),

            "rep_dtype": "float", # If this changes, also need to modify the arithmetic intensity calculation
            "arithmetic_intensity (FLOPs / byte)": ops_per_tp * batch_size / total_data_streamed, 

            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps_rough": bandwidth_gbps_rough
        }
        logger.info(f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")

        return result