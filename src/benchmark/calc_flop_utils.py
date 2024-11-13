import math

from src.implementations.e3nn_lite import TPProblem
from src.benchmark.logging_utils import getLogger
from src.benchmark.e3nn_lite_utils import sparse_outer_product_work, count_cg_non_zero, cg

logger = getLogger()

def calculate_minimum_flops_forward(tpp : TPProblem, batch_size : int) -> dict:
    """
    This is not actually calcuating the minumum value. 
    Ideally you might share the outer product values between two inputs across multiple inputs. 
    This is assuming that you form those values and reuse them once per CG decomp.
    """
    logger.warning("This is not accurately calculating the minimum amount of flops that can be performed")
    flops_count = {}
    flops_count["outer_products"] = 0
    flops_count["CG_decomposition"] = 0
    flops_count["linear_combination"] = 0
    for ins in tpp.instructions: # type : Instruction
        l1, l2, l3 = tpp.irreps_in1[ins.i_in1].ir.l, tpp.irreps_in2[ins.i_in2].ir.l, tpp.irreps_out[ins.i_out].ir.l
        assert isinstance(l1, int)
        assert isinstance(l2, int)
        assert isinstance(l3, int)
        
        flops_count["outer_products"] += sparse_outer_product_work(cg(l1,l2,l3))
        flops_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (ins.path_shape[0] * ins.path_shape[1])
        flops_count["linear_combination"] += (2 * l3 + 1) * math.prod(ins.path_shape) if ins.has_weight else 0

    flops_count["outer_products"] *= batch_size
    flops_count["CG_decomposition"] *= batch_size
    flops_count["linear_combination"] *= batch_size

    flops_count["total"] = sum(flops_count.values())
    return flops_count


def calculate_minimum_flops_backward(tpp : TPProblem, batch_size : int) -> dict:
    """
    This is not actually calcuating the minumum value. 
    Ideally you might share the outer product values between two inputs across multiple inputs. 
    This is assuming that you form those values and reuse them once per CG decomp.
    """
    raise NotImplementedError("this needs to be implemented properly")