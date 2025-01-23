import numpy as np
import numpy.linalg as la

from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.implementations.convolution.Convolution import *
from openequivariance.benchmark.tpp_creation_utils import *

class CUEConv(Convolution):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        super().__init__(config, idx_dtype, torch_op)

        self.reference_tp = CUETensorProduct(config, torch_op)
        self.cue_tp = self.reference_tp.cue_tp

        from openequivariance.implementations.convolution.scatter import scatter_sum
        self.scatter_sum = scatter_sum

    @staticmethod
    def name():
        return "CUEConvolution"

    def forward(self, L1_in, L2_in, weights, rows, cols):
        tp_outputs = self.cue_tp(L1_in[cols], L2_in, weights, use_fallback=False)
        return self.scatter_sum(src=tp_outputs, index=rows, dim=0, dim_size=L1_in.shape[0])