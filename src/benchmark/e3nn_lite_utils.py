import functools
import warnings
import math

import numpy as np 

from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import Irrep, _MulIr, Irreps, Instruction, TPProblem

def sparse_outer_product_work(cg : np.ndarray) -> int: 
    return np.sum(np.max(cg != 0, axis=2))

def convenience_namer(L1 : Irreps, L2 : Irreps, L3 : Irreps):
    return f"({L1}x{L2}->{L3})"

# Non Zeros 
@functools.lru_cache(typed=True)
def count_cg_non_zero(l1, l2, l3) -> int:
    return np.count_nonzero(TensorProduct.load_cg_tensor(l1, l2, l3))

def calculate_total_nnz(tpp : TPProblem) -> int:
        """
        To make sure you don't over count repeat CGs which get used multiple times 
        """
        nnz_by_l_combo = {}
        for ins in tpp.instructions: # type : Instruction
            l1 = tpp.irreps_in1[ins.i_in1].ir.l 
            l2 = tpp.irreps_in2[ins.i_in2].ir.l
            l3 = tpp.irreps_out[ins.i_out].ir.l
            assert isinstance(l1, int)
            assert isinstance(l2, int)
            assert isinstance(l3, int)
            nnz_by_l_combo[(l1,l2,l3)] =  count_cg_non_zero(l1,l2,l3)
        return sum(nnz_by_l_combo.values())

def calc_weight_offsets(tpp : TPProblem) -> list[int]:
    """
    Returns a list of weight offsets for every instruction. 
    """
    assert isinstance(tpp, TPProblem)
    offset = 0
    offsets = []
    for ins in tpp.instructions:
        assert isinstance(ins, Instruction) 
        offsets.append(offset)
        if ins.has_weight:
            flatsize = math.prod(ins.path_shape)
            offset += flatsize
    return offsets     

class RepData:
    def __init__(self, irreps : Irreps):
        assert isinstance(irreps, Irreps)
        self.rep_len = irreps.dim
        self.ls = [mul_irrep.ir.l for mul_irrep in irreps]
        self.irrep_lengths = [mul_irrep.ir.dim for mul_irrep in irreps]
        self.mults = [mul_irrep.mul for mul_irrep in irreps]

        offset = 0
        self.offsets = []
        for mul_irrep in irreps: 
            self.offsets.append(offset)
            offset += mul_irrep.dim

class CGTensor:
    def __init__(self, l1, l2, l3):
        tensor = TensorProduct.load_cg_tensor(l1, l2, l3)
        coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
        float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        values = [str(float.hex(float(val))) + "f" for val in float_values]
        self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
        self.nnz = len(values)