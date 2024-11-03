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
    next_tp_id = 0 # Used to assign unique IDs to each TP instance 
    tensors = None
    with open(pathlib.Path("data/CG_tensors.pickle"), 'rb') as f:
        tensors = pickle.load(f) 

    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely.
    '''
    def __init__(self, config, torch_op=False):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out

        self.tp_id = TensorProduct.next_tp_id
        TensorProduct.next_tp_id += 1

        if torch_op:
            self.setup_torch_module()

    @staticmethod
    def name():
        raise NotImplementedError()

    def exec_tensor_product(self,
            batch : np.uint64,
            L1_in: np.uint64,
            L2_in: np.uint64,
            L3_out: np.uint64,
            weights: np.uint64):
        '''
        Inputs are integers representing device pointers.
        '''
        self.internal.exec_tensor_product(batch, L1_in, L2_in, L3_out, weights) 

    def exec_tensor_product_cpu(self, 
        L1_in: np.ndarray, 
        L2_in: np.ndarray, 
        L3_out: np.ndarray, 
        weights: np.ndarray):
        '''
        All state initialization for the internal class occurs inside the
        constructor. 
        '''
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)

    def backward(self, batch_size: np.uint64,
                L1_in: np.uint64, L1_grad: np.uint64, 
                L2_in: np.uint64, L2_grad: np.uint64,
                weights: np.uint64, weights_grad: np.uint64,
                L3_grad: np.uint64):
        '''
        Inputs are integers representing device pointers.
        '''
        self.internal.backward(
                batch_size,
                L1_in, L1_grad,
                L2_in, L2_grad,
                weights, weights_grad,
                L3_grad)

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

    def test_correctness(self, L1_in, L2_in, weights, L3_out_comp,
            reference_implementation):
        L1, L2, L3 = self.L1, self.L2, self.L3
        config = self.config 

        thresh = 5e-7
        result = {
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": thresh, # Above floating point interval machine epsilon 
            "pass": False
        }

        ground_truth = np.zeros((L1_in.shape[0], L3.dim), dtype=np.float32)
        tp = reference_implementation(self.config)
        tp.exec_tensor_product_cpu(L1_in, L2_in, ground_truth, weights)

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
                    L1_in, L2_in, L3_buffer, weights,
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
        interactions = [(u, v, w) for (u, v, w, *others) in self.config.instructions] 

        L1_in  = np.array(rng.uniform(size=(batch_size, self.L1.dim)), dtype=np.float32) 
        L2_in  = np.array(rng.uniform(size=(batch_size, self.L2.dim)), dtype=np.float32)
        L3_buffer = np.zeros((batch_size, self.L3.dim), dtype=np.float32)
        weights = np.array(rng.uniform(size=(batch_size, self.config.weight_numel)), dtype=np.float32)

        L1_grad, L2_grad, weights_grad = [None] * 3
        if direction == "backward":
            L3_buffer[:] = rng.uniform(size=(batch_size, L3.dim)) 
            L1_grad = np.zeros_like(L1_in)
            L2_grad = np.zeros_like(L2_in)
            weights_grad = np.zeros_like(weights)

        logger.info("Initialized input / output data.")

        # Forward: Requires two multiplications and one addition --> 3, 4 if weights are included (not yet)
        # Backward: Requires 6 multiplications and 3 additions (including the weight, implemented)
        ops_per_nz, total_data_streamed  = None, None
        if direction == "forward":
            ops_per_nz = 3
            total_data_streamed = L1_in.nbytes + L2_in.nbytes + L3_buffer.nbytes + weights.nbytes 
        elif direction == "backward":
            ops_per_nz = 9
            total_data_streamed = L1_in.nbytes + L2_in.nbytes + L3_buffer.nbytes + weights.nbytes \
                    + L1_grad.nbytes + L2_grad.nbytes + weights_grad.nbytes

        ops_per_tp = 0
        nnz = 0
        for u, v, w in interactions:
            tensor = self.load_cg_tensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l)
            local_nnz = np.count_nonzero(tensor)
            nnz += local_nnz
            ops_per_tp += ops_per_nz * local_nnz * L1[u].mul * L2[v].mul # Assumes L3.mult(w) = L1.mult(u) * L2.mult(v) 
            ops_per_tp += L3[w].mul * (2 * L3[w].ir.l + 1) # FLOPS for weights, assuming "uvu" 

        # =========== Benchmarking ===========
        time_millis = self.benchmark_internal(num_warmup, num_iter, L1_in, L2_in, L3_buffer, weights, L1_grad, L2_grad, weights_grad, direction)
        # ==================================== 

        # We don't multiply by num_iters since we benchmark each kernel run separately 
        throughputs_gflops = [float(el) for el in batch_size * ops_per_tp / (time_millis * 1e6)]
        bandwidth_gbps_rough = [float(el) for el in total_data_streamed / (time_millis * 1e6)]
        time_millis = [float(el) for el in time_millis] 

        result = {
            "tp_direction": direction,
            "total_cg_nnz": nnz,
            "flops_per_tp": ops_per_tp,
            "L1": str(L1),
            "L2": str(L2), 
            "L3": str(L3),
            "instructions": self.config.instructions, 

            "L1_rep_len": L1.dim,
            "L2_rep_len": L2.dim,
            "L3_rep_len": L3.dim,

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

    def setup_torch_module(self):
        import torch, typing

        # ----------------- Forward pass -----------------
        @torch.library.custom_op(f"fast_tp::tp_forward{self.tp_id}", mutates_args=(), device_types="cuda")
        def forward(L1_in : torch.Tensor, L2_in : torch.Tensor, weights : torch.Tensor) -> torch.Tensor:
            L1_in_c, L2_in_c, weights_c = L1_in.contiguous(), L2_in.contiguous(), weights.contiguous()
            L3_out = torch.zeros((L1_in_c.shape[0], self.reps.L3.get_rep_length() ), dtype=torch.float32, device='cuda')
            self.exec_tensor_product(L1_in_c.shape[0], L1_in_c.data_ptr(), L2_in_c.data_ptr(), L3_out.data_ptr(), weights_c.data_ptr())
            return L3_out
        
        @forward.register_fake
        def _(L1_in, L2_in, weights):
            return L1_in.new_empty(L1_in.shape[0], self.reps.L3.get_rep_length())
        
        self.forward = forward
        
        # ---------------- Backward pass -----------------
        @torch.library.custom_op(f"fast_tp::tp_grad_helper{self.tp_id}", mutates_args=(), device_types="cuda")
        def grad_helper( L1_in : torch.Tensor, L2_in : torch.Tensor, 
                     weights : torch.Tensor, L3_grad : torch.Tensor ) -> typing.List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.zeros_like(L2_in)
            weights_grad = torch.zeros_like(weights)
            
            self.backward( L1_in.shape[0], L1_in.data_ptr(), L1_grad.data_ptr(),
                        L2_in.data_ptr(), L2_grad.data_ptr(),
                        weights.data_ptr(), weights_grad.data_ptr(),
                        L3_grad.data_ptr() )
            
            return [L1_grad, L2_grad, weights_grad]
        
        @grad_helper.register_fake
        def _(L1_in, L2_in, weights, L3_grad):
            return [L1_in.new_empty(*L1_in.shape), L2_in.new_empty(*L2_in.shape), weights.new_empty(*weights.shape)]

        def setup_context(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights = inputs
        
        def backward(ctx, grad_output):
            result = grad_helper(ctx.L1_in, ctx.L2_in, ctx.weights, grad_output)
            return result[0], result[1], result[2]

        self.forward.register_autograd(backward, setup_context=setup_context)

        # Setup for higher derivatives
        def setup_context_grad_helper(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights, ctx.L3_grad = inputs 

        def grad_helper_backward(ctx, grad_output):
            A, B, C, D = ctx.L1_in, ctx.L2_in, ctx.L3_grad, ctx.weights
            E, F, G = grad_output[0], grad_output[1], grad_output[2]

            op1 = grad_helper(A, B, D, C)
            op2 = grad_helper(A, B, G, C)
            op3 = forward(E, B, D)
            op4 = grad_helper(E, B, D, C) # op4 and op5 could be combined with op3 and op6 
            op5 = grad_helper(A, F, D, C) 
            op6 = forward(A, F, D)
            op7 = forward(A, B, G)

            return op1[0] + op2[0], op1[1] + op2[1], op4[2] + op5[2], op3 + op6 + op7

        grad_helper.register_autograd(grad_helper_backward, setup_context=setup_context_grad_helper)