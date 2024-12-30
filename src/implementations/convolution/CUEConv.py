import numpy as np
import numpy.linalg as la

from src.implementations.convolution.Convolution import *
from src.implementations.E3NNTensorProduct import *
from src.implementations.convolution.scatter import scatter_sum
from src.benchmark.tpp_creation_utils import *

class CUEConv(Convolution):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        assert(torch_op)
        super().__init__(config, idx_dtype, torch_op)

        assert(not config.shared_weights)

        global torch
        import torch
        import cuequivariance as cue
        import cuequivariance_torch as cuet

        supported_tpp_types = [
            ChannelwiseTPP,
            FullyConnectedTPProblem,
            SingleInstruction
        ]

        assert(config.irrep_dtype == config.weight_dtype)

        np_to_torch_dtype = {
            np.float32: torch.float32,
            np.float64: torch.float64
        }

        assert(any([isinstance(config, supported_ttp_type)] for supported_ttp_type in supported_tpp_types))
        if isinstance(config, ChannelwiseTPP) or isinstance(config, SingleInstruction):
            e = cue.descriptors.channelwise_tensor_product(
                cue.Irreps("O3", str(config.irreps_in1)),
                cue.Irreps("O3", str(config.irreps_in2)),
                cue.Irreps("O3", str(config.irreps_out)))
        
        if isinstance(config, FullyConnectedTPProblem):
            e = cue.descriptors.fully_connected_tensor_product(
                cue.Irreps("O3", str(config.irreps_in1)),
                cue.Irreps("O3", str(config.irreps_in2)),
                cue.Irreps("O3", str(config.irreps_out)),
            )

        assert(config.weight_numel == e.inputs[0].irreps.dim)
        self.cue_tp = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul, math_dtype=np_to_torch_dtype[config.irrep_dtype])        
        self.cue_tp.to('cuda')

    @staticmethod
    def name():
        return "CUEConvolution"

    def forward(self, L1_in, L2_in, weights, src, dst):
        tp_outputs = self.cue_tp(weights, L1_in[src], L2_in, use_fallback=False)
        return scatter_sum(src=tp_outputs, index=dst, dim=0, dim_size=L1_in.shape[0])
