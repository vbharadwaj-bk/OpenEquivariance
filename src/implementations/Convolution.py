import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.benchmark.logging_utils import getLogger, bcolors 
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
    def __init__(self, io_reps: RepTriple):
        self.io_reps = io_reps
        self.L1, self.L2, self.L3 = io_reps.L1, io_reps.L2, io_reps.L3
        self.internal = None

    @staticmethod
    def name():
        raise NotImplementedError()

    def exec_conv_cpu(self, 
            L1_in, L2_in, L3_out,
            graph, disable_tensor_op=False):
        self.internal.exec_conv_cpu(L1_in, L2_in, L3_out, 
                graph.coords, graph.rows, graph.cols, 
                disable_tensor_op)

    def test_correctness_no_op(self, L1_in, L2_in, L3_out_comp, graph, reuse_cached_graph=False):
        '''
        Tests correctness by performing a "no-op" tensor product. For
        each nonzero (i, j), A[i:] += B[j:]. This test requires
        the input and output reps to have the same length; edge features
        are ignored. 
        '''
        L1, L2, L3 = self.L1, self.L2, self.L3
        assert(L1.get_rep_length() == L3.get_rep_length())

        from scipy.sparse import csr_matrix

        logger.info("Starting reference SpMM for convolution...")
        if not reuse_cached_graph or graph.cached_sp_graph is None:
            graph.cached_sp_graph = csr_matrix((np.ones(len(graph.rows)), (graph.rows, graph.cols)), shape=(graph.node_count, graph.node_count))
            
        ground_truth = graph.cached_sp_graph @ L1_in
        logger.info("Finished reference SpMM.")

        thresh = 5e-6 # AtomicAdd nondeterminism may require higher threshold 
        result = {
            "disable_tensor_op": True,
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
                logger.info(f"{bcolors.OKGREEN}No-tensor-op convolution correctness check pass, {diff_Linf_norm=:.2g}, {thresh=:.2g}. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}No-tensor-op convolution correctness check fail! {diff_Linf_norm=:.2g}, {thresh=:.2g} {bcolors.ENDC}")

        return result, ground_truth

    def benchmark_internal(self, num_warmup, num_iter, L1_in, L2_in, L3_out, graph, disable_tensor_op):
        '''
        Returns the total time for num_iter iterations of the core inner loop
        after num_warmup warmup iterations. Can override for other implementations
        '''
        time_millis = np.zeros(num_iter, dtype=np.float32)

        self.internal.benchmark_cpu(L1_in, L2_in, L3_out, 
                graph.coords, graph.rows, graph.cols, 
                disable_tensor_op,
                num_warmup,
                time_millis)

        return time_millis

    def benchmark(self, num_warmup, num_iter, graph, disable_tensor_op, prng_seed=12345):
        '''
        This function only works for scalar L-values right now, need to change
        to handle any multiplicity.
        '''
        rng = np.random.default_rng(prng_seed)

        L1_in  = np.array(rng.uniform(size=(graph.node_count, self.L1.get_rep_length())), dtype=np.float32) 
        L2_in  = np.array(rng.uniform(size=(graph.node_count, self.L2.get_rep_length())), dtype=np.float32)
        L3_out = np.zeros((graph.node_count, self.L3.get_rep_length()), dtype=np.float32)

        L1, L2, L3 = self.L1, self.L2, self.L3

        #assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        #cg_tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        #nnz = len(np.nonzero(cg_tensor)[0])

        # =========== Benchmarking ===========
        time_millis = self.benchmark_internal(num_warmup, num_iter, L1_in, L2_in, L3_out, graph, disable_tensor_op)
        # ==================================== 

        if disable_tensor_op:
            throughputs_gflops = [float(el) for el in graph.nnz * self.L1.get_rep_length() / (time_millis * 1e6)]

            # Rough calculation of bandwidth assumes output is touched only once, but input rows are read as many times as nnz 
            bandwidth_gbps_rough = [float(el) for el in (L3_out.nbytes + L1_in[0, :].nbytes * graph.nnz) / (time_millis * 1e6)]
            time_millis = [float(el) for el in time_millis] 
        else:
            raise NotImplementedError("No throughput / bwidth calculation implemented when tensor op is enabled!")

        result = {
            "disable_tensor_op": disable_tensor_op, 
            "L1": L1.to_string(),
            "L2": L2.to_string(),
            "L3": L3.to_string(),
            "graph_node_count": graph.node_count,
            "graph_adj_nnz": graph.nnz,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps_rough": bandwidth_gbps_rough
        }

        disable_op_str = ""
        if disable_tensor_op:
            disable_op_str = " (Tensor Op Disabled)"

        logger.info(f"{bcolors.OKCYAN}Avg. Throughput{disable_op_str}: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")
        logger.info(f"{bcolors.OKCYAN}Avg. Bandwidth{disable_op_str}: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(bandwidth_gbps_rough):.2f} ± {np.std(bandwidth_gbps_rough):.2f} GBPs{bcolors.ENDC}")
        return result

