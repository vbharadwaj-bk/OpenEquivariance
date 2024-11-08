import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.benchmark.logging_utils import getLogger, bcolors 
from src.implementations.TensorProduct import flops_data_per_tp
logger = getLogger()

class CoordGraph:
    def __init__(self, coords, rows, cols, name):
        '''
        Because graphs may change constantly, this class is designed
        to be as light as possible. A directed edge from node
        u to v is indicated by the presence of an index i such that
        rows[i] = u, rows[i] = v.
        '''
        assert(len(rows) == len(cols))
        self.nnz = len(rows) # Counts every nonzero in the adjacency matrix 
        self.node_count = coords.shape[0]
        self.rows = rows
        self.cols = cols
        self.coords = coords
        self.name = name

        self.cached_sp_graph = None # Cached scipy sparse matrix 

class Convolution:
    '''
    Inputs: L1 for input node features
            L2 for edge features
            L3 for output node features 
    '''
    def __init__(self, config):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.internal = None

    @staticmethod
    def name():
        raise NotImplementedError()

    def exec_conv_cpu(self, 
            L1_in, L2_in, weights, L3_out,
            graph, disable_tensor_op=False):
        self.internal.exec_conv_cpu(L1_in, L2_in, weights, L3_out, 
                graph.rows, graph.cols, disable_tensor_op)

    def backward_cpu(self, 
            L1_in, L2_in, weights, L3_grad,
            graph, disable_tensor_op=False):
        '''
        We break from convention here by allocating and returning
        the appropriate buffers. 
        '''
        L1_grad = np.zeros_like(L1_in)
        L2_grad = np.zeros_like(L2_in)
        weights_grad = np.zeros_like(weights)

        self.internal.backward_cpu(
            L1_in, L1_grad,
            L2_in, L2_grad,
            weights, weights_grad,
            L3_grad,
            graph.rows, graph.cols,
            disable_tensor_op)

        return L1_grad, L2_grad, weights_grad

    def test_correctness(self, L1_in, L2_in, weights, L3_out_comp, graph, conv_reference_impl, disable_tensor_op):
        L1, L2, L3 = self.L1, self.L2, self.L3

        ground_truth = np.zeros((graph.node_count, L3.dim), dtype=np.float32)
        conv_reference = conv_reference_impl(self.config)

        if disable_tensor_op:
            logger.warning(f"{bcolors.WARNING}Tensor product disabled within convolution, performing SpMM.{bcolors.ENDC}")

        logger.info(f"Starting reference convolution {bcolors.OKCYAN}{conv_reference.name()}{bcolors.ENDC}.")
        conv_reference.exec_conv_cpu(L1_in, L2_in, weights, ground_truth, graph, disable_tensor_op) 
        logger.info("Finished reference convolution.")

        thresh = 5e-6 # AtomicAdd nondeterminism may require higher threshold 
        result = {
            "disable_tensor_op": disable_tensor_op,
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": thresh, # Above floating point interval machine epsilon 
            "pass": False
        }

        if L3_out_comp.shape != ground_truth.shape:
            result["shape_match"] = False
            logger.error(f"{bcolors.FAIL}Ground truth shape does not match input! {L3_out_comp.shape=}, {ground_truth.shape=} {bcolors.ENDC}")
        else:
            result["shape_match"] = True 
            diff_Linf_norm = float(la.norm((ground_truth - L3_out_comp).flatten(), ord=np.inf))
            result["diff_Linf_norm"] = diff_Linf_norm 
            result["pass"] = bool(diff_Linf_norm < thresh) 

            if result["pass"]:
                logger.info(f"{bcolors.OKGREEN}Convolution correctness check pass, {diff_Linf_norm=:.2g}, {thresh=:.2g}. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}Convolution correctness check fail! {diff_Linf_norm=:.2g}, {thresh=:.2g} {bcolors.ENDC}")

        return result, ground_truth

    def benchmark_internal(self, num_warmup, num_iter, L1_in, L2_in, weights, L3_out, graph, disable_tensor_op):
        '''
        Returns the total time for num_iter iterations of the core inner loop
        after num_warmup warmup iterations. Can override for other implementations
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        self.internal.benchmark_cpu(L1_in, L2_in, weights, L3_out, 
                graph.coords, graph.rows, graph.cols, 
                disable_tensor_op,
                num_warmup,
                time_millis)

        return time_millis

    def benchmark_internal(self, num_warmup, num_iter, L1_in, L2_in, L3_buffer, weights, 
            L1_grad, L2_grad, weights_grad, direction, graph, disable_tensor_op):

        time_millis = np.zeros(num_iter, dtype=np.float32)

        if direction == "forward":
            self.internal.benchmark_forward_cpu(L1_in, L2_in, weights, L3_buffer, 
                    graph.coords, graph.rows, graph.cols, 
                    disable_tensor_op,
                    num_warmup,
                    time_millis)
        
        elif direction == "backward":
            self.internal.benchmark_backward_cpu(
                L1_in, L1_grad,
                L2_in, L2_grad,
                weights, weights_grad,
                L3_buffer,
                graph.rows, graph.cols,
                disable_tensor_op,
                num_warmup,
                time_millis)

        return time_millis

    def benchmark(self, num_warmup, num_iter, graph, disable_tensor_op, direction, prng_seed=12345):
        '''
        This function only works for scalar L-values right now, need to change
        to handle any multiplicity.
        '''
        L1, L2, L3, config = self.L1, self.L2, self.L3, self.config
        rng = np.random.default_rng(prng_seed)

        L1_in  = np.array(rng.uniform(size=(graph.node_count, L1.dim)), dtype=np.float32)
        L2_in  = np.array(rng.uniform(size=(graph.nnz, L2.dim)), dtype=np.float32)
        weights = np.array(rng.uniform(size=(graph.nnz, config.weight_numel)), dtype=np.float32)
        L3_buffer = np.zeros((graph.node_count, L3.dim), dtype=np.float32)

        L1_grad, L2_grad, weights_grad = [None] * 3
        if direction == "backward":
            L3_buffer[:] = rng.uniform(size=(graph.node_count, L3.dim)) 
            L1_grad = np.zeros_like(L1_in)
            L2_grad = np.zeros_like(L2_in)
            weights_grad = np.zeros_like(weights)

        # =========== Benchmarking ===========
        time_millis = self.benchmark_internal(num_warmup, num_iter, L1_in, L2_in, L3_buffer, weights, 
            L1_grad, L2_grad, weights_grad, direction, graph, disable_tensor_op)
        # ==================================== 

        ops_per_tp, data_per_tp, nnz = flops_data_per_tp(self.config, 4, "forward")
        if disable_tensor_op:
            ops_per_tp = 2 * L3.dim
        else:
            ops_per_tp += L3.dim # Output accumulation 

        throughputs_gflops = [float(el) for el in graph.nnz * ops_per_tp / (time_millis * 1e6)]

        # Rough calculation of bandwidth assumes output is touched only once, but input rows are read as many times as nnz 
        bandwidth_gbps = [float(el) for el in graph.nnz * data_per_tp / (time_millis * 1e6)]
        time_millis = [float(el) for el in time_millis] 

        result = {
            "disable_tensor_op": disable_tensor_op, 
            "L1": str(L1),
            "L2": str(L2), 
            "L3": str(L3),
            "graph_node_count": graph.node_count,
            "graph_adj_nnz": graph.nnz,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps": bandwidth_gbps
        }

        disable_op_str = ""
        if disable_tensor_op:
            disable_op_str = " (Tensor Op Disabled)"

        logger.info(f"{bcolors.OKCYAN}Avg. Throughput{disable_op_str}: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")
        logger.info(f"{bcolors.OKCYAN}Avg. Bandwidth{disable_op_str}: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(bandwidth_gbps):.2f} ± {np.std(bandwidth_gbps):.2f} GBPs{bcolors.ENDC}")
        return result