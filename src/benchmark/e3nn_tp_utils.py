import functools
import warnings
import math

import numpy as np 

import e3nn
from e3nn.o3 import Irrep, Irreps, Instruction

def sparse_outer_product_work(cg : np.ndarray) -> int: 
    return np.sum(np.max(cg != 0, axis=2))

def cg(l1 : int, l2 : int, l3 : int) -> np.ndarray:
    return e3nn.o3.wigner_3j(l1, l2, l3).numpy(force=True)

def convenience_namer(L1 : Irreps, L2 : Irreps, L3 : Irreps):
    return f"({L1}x{L2}->{L3})"

def get_random_forward_supplies(e3nn_tp : e3nn.o3.TensorProduct, batch_size : int,  prng_seed : int = 12345 ) -> tuple[np.ndarray]:
    """
    Return properly sized numpy arrays needed to execute a tensor product in the forward direction
    """
    rng = np.random.default_rng(prng_seed)

    in1 = np.array(rng.uniform(size=(batch_size, e3nn_tp.irreps_in1.dim)), dtype=np.float32) 
    in2 = np.array(rng.uniform(size=(batch_size, e3nn_tp.irreps_in2.dim)), dtype=np.float32)
    weights = np.array(rng.uniform(size=(batch_size, e3nn_tp.weight_numel)), dtype=np.float32)

    out = np.zeros(size=(batch_size, e3nn_tp.irreps_out.dim), dtype=np.float32)

    return in1, in2, weights, out 

def get_random_backward_supplies(e3nn_tp : e3nn.o3.TensorProduct, batch_size : int, prng_seed : int = 12345) -> tuple[np.ndarray]:
    """
    Return properly sized numpy arrays needed to execute a tensor product in the backward direction
    """
    rng = np.random.default_rng(prng_seed)
    
    in1 = np.array(rng.uniform(size=(batch_size, e3nn_tp.irreps_in1.dim)), dtype=np.float32) 
    in2 = np.array(rng.uniform(size=(batch_size, e3nn_tp.irreps_in2.dim)), dtype=np.float32)
    out_grad = np.array(rng.uniform(size=(batch_size, e3nn_tp.irreps_out.dim)), dtype=np.float32)
    weights = np.array(rng.uniform(size=(batch_size, e3nn_tp.weight_numel)), dtype=np.float32)

    weights_grad = np.zeros_like(weights)
    in1_grad = np.zeros_like(in1)
    in2_grad = np.zeros_like(in2) 

    return in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad

# Non Zeros 
@functools.lru_cache(typed=True)
def count_cg_non_zero(l1, l2, l3) -> int:
    return np.count_nonzero(cg(l1,l2,l3))

def calculate_total_nnz( e3nn_tp : e3nn.o3.TensorProduct) -> int:
        """
        To make sure you don't over count repeat CGs which get used multiple times 
        """
        nnz_by_l_combo = {}
        for (l1, _, l2, _, l3, _) in e3nn_tp.paths.keys():
            assert isinstance(l1, int)
            assert isinstance(l2, int)
            assert isinstance(l3, int)
            nnz_by_l_combo[(l1,l2,l3)] =  count_cg_non_zero(l1,l2,l3)
        return sum(nnz_by_l_combo.values())

def calculate_minimum_memory_streamed_forward(e3nn_tp : e3nn.o3.TensorProduct, batch_size : int) -> dict:
    data_size = {}
    data_size["input"]   = (e3nn_tp.irreps_in1.dim * e3nn_tp.irreps_in2.dim) * batch_size
    data_size["output"]  = (e3nn_tp.irreps_out.dim) * batch_size
    data_size["weights"] = e3nn_tp.weight_numel
    data_size["total"] = sum(data_size.values())
    return data_size        

def calculate_minimum_flops_forward(e3nn_tp : e3nn.o3.TensorProduct, batch_size : int) -> dict:
    """
    This is not actually calcuating the minumum value. 
    Ideally you might share the outer product values between two inputs across multiple inputs. 
    This is assuming that you form those values and reuse them once per CG decomp.
    """
    warnings.warn(message="This is not acurately calculating the minimum amount of flops that can be performed")
    flops_count = {}
    flops_count["outer_products"] = 0
    flops_count["CG_decomposition"] = 0
    flops_count["linear_combination"] = 0
    for ins in e3nn_tp.instructions: # type : Instruction
        l1, l2, l3 = e3nn_tp.irreps_in1[ins.i_in1].l, e3nn_tp.irreps_in2[ins.i_in2], e3nn_tp.irreps_out[ins.i_out]
        flops_count["outer_products"] += sparse_outer_product_work(cg(l1,l2,l3))
        flops_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (ins.path_shape[0] * ins.path_shape[1])
        flops_count["linear_combination"] += (2 * l3 + 1) * math.prod(ins.path_shape) if ins.has_weight else 0

    flops_count["outer_products"] *= batch_size
    flops_count["CG_decomposition"] *= batch_size
    flops_count["linear_combination"] *= batch_size

    flops_count["total"] = flops_count["outer_products"] + flops_count["CG_decomposition"] + flops_count["linear_combination"]
    return flops_count

def calc_weight_offsets(e3nn_tp : e3nn.o3.TensorProduct):
    """
    Returns a list of weight offsets for every instruction. 
    """
    assert isinstance(e3nn_tp, e3nn.o3.TensorProduct)
    offset = 0
    offsets = []
    for ins in e3nn_tp.instructions:
        assert isinstance(ins, Instruction) 
        offsets.append(offset)
        if ins.has_weight:
            flatsize = math.prod(ins.path_shape)
            offset += flatsize
    return offsets     
        