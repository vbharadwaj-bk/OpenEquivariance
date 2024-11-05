import numpy as np
import numpy.linalg as la

from src.implementations.Convolution import *
from src.implementations.NumpyTensorProduct import *

class NumpyConv(Convolution):
    def __init__(self, config):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.reference_tp = NumpyTensorProduct(config)

    @staticmethod
    def name():
        return "NumpyConvolution" 

    def exec_conv_cpu(self, 
            L1_in, L2_in, weights, L3_out,
            graph, disable_tensor_op=False):
        '''
        L1_in:   [nodes, features_L1]
        L2_in:   [edges, features_L2]
        weights: [edges, weight_numel]
        L3_out:  [nodes, features_L3]
        '''
        if disable_tensor_op:
            assert(L1_in.shape[1] == L3_out.shape[1])
            np.add.at(L3_out, graph.rows, L1_in[graph.cols])
        else:
            tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=np.float32)
            self.reference_tp.exec_tensor_product_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
            np.add.at(L3_out, graph.rows, tp_outputs)