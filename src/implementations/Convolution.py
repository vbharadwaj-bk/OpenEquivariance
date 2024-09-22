import pickle, pathlib
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

class CoordGraph:
    def __init__(self, coords, rows, cols):
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

        thresh = 5e-7
        result = {
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
                logger.info(f"{bcolors.OKGREEN}No-tensor-op convolution correctness check pass. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}No-tensor-op convolution correctness check fail! {diff_Linf_norm=}, {thresh=} {bcolors.ENDC}")

        return result, ground_truth

