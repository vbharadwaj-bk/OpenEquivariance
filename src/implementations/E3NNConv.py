import numpy as np
import numpy.linalg as la

from src.implementations.Convolution import *
from src.implementations.E3NNTensorProduct import *

class E3NNConv(Convolution):
    def __init__(self, config):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out

        global torch
        import torch
        import e3nn

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float64)

        self.e3nn_tp = e3nn.o3.TensorProduct(
                    config.irreps_in1, 
                    config.irreps_in2, 
                    config.irreps_out, 
                    config.instructions_raw,
                    in1_var=config.in1_var,
                    in2_var=config.in2_var,
                    out_var=config.out_var,
                    irrep_normalization=config.irrep_normalization,
                    path_normalization=config.path_normalization,
                    internal_weights=config.internal_weights,
                    shared_weights=config.shared_weights).to(device='cuda')

        self.reference_tp = E3NNTensorProduct(config)

        if config.irrep_dtype == np.float64:
            torch.set_default_dtype(torch.float32)  # Reset to default


    @staticmethod
    def name():
        return "E3NNConvolution" 

    def forward_cpu(self, 
            L1_in, L2_in, weights, L3_out,
            graph, disable_tensor_op=False):
        disable_tensor_op = False

        tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=np.float32)
        self.reference_tp.forward_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
        np.add.at(L3_out, graph.rows, tp_outputs)