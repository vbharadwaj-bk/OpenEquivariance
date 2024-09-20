import pickle, pathlib
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

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
        self.cached_sp_graph = None

    @staticmethod
    def name():
        raise NotImplementedError()

    def exec_conv_cpu(self, 
            L1_in, L2_in, L3_out,
            rows, cols, no_tensor_op=False):
        self.internal.exec_conv_cpu(L1_in, L2_in, L3_out, rows, cols, no_tensor_op)

    def test_correctness_no_op(self, L1_in, L2_in, L3_out_comp, 
            rows, cols, reuse_cached_graph=False):
        '''
        Tests correctness by performing a "no-op" tensor product. For
        each nonzero (i, j), A[i:] += B[j:]. This test requires
        the input and output reps to have the same length; edge features
        are ignored. 
        '''
        L1, L2, L3 = self.L1, self.L2, self.L3
        assert(L1.get_rep_length() == L3.get_rep_length())

        from scipy.sparse import csr_matrix

        if not reuse_cached_graph or self.cached_sp_graph is None:
            self.cached_sp_graph = csr_matrix( (np.ones(len(rows)), rows, cols) )

        ground_truth = self.cached_sp_graph @ L1_in

        thresh = 5e-7
        result = {
            "shape_match": False,
            "diff_Linf_norm": np.inf,
            "thresh": thresh, # Above floating point interval machine epsilon 
            "pass": False
        }

        if L3_out_comp.shape != ground_truth.shape:
            result["shape_match"] = False
            logger.error(f"{bcolors.FAIL}Ground truth shape does not match input! {diff_Linf_norm=}, {thresh=} {bcolors.ENDC}")
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

