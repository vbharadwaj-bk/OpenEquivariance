import torch
import e3nn

import numpy as np
import numpy.linalg as la

from src.implementations.Convolution import *
from src.implementations.E3NNTensorProduct import *

'''
Scatter sum operator from MACE.

basic scatter_sum operations from torch_scatter from
https://github.com/mir-group/pytorch_runstats/blob/main/torch_runstats/scatter_sum.py
Using code from https://github.com/rusty1s/pytorch_scatter, but cut down to avoid a dependency.
'''
def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

class E3NNConv(Convolution):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        assert(torch_op)
        super().__init__(config, idx_dtype, torch_op)

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

    def forward(self, L1_in, L2_in, weights, src, dst):
        tp_outputs = self.reference_tp(L1_in[src], L2_in, weights)
        return scatter_sum(src=tp_outputs, index=dst, dim=0, dim_size=L1_in.shape[0])

    @staticmethod
    def name():
        return "E3NNConvolution" 

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=L3_out.dtype)
        self.reference_tp.forward_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
        np.add.at(L3_out, graph.rows, tp_outputs)

    def backward_cpu(
            self,
            L1_in : np.ndarray,
            L1_grad : np.ndarray,
            L2_in : np.ndarray,
            L2_grad : np.ndarray,
            L3_grad : np.ndarray,
            weights : np.ndarray,
            weights_grad : np.ndarray,
            graph):
        L1_grad_bcast = np.zeros((graph.nnz, self.L1.dim), dtype=L1_grad.dtype)
        self.reference_tp.backward_cpu(
                L1_in[graph.cols], L1_grad_bcast, L2_in, L2_grad, L3_grad[graph.rows], weights, weights_grad)
        np.add.at(L1_grad, graph.cols, L1_grad_bcast)