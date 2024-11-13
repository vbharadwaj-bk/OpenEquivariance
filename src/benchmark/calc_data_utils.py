from src.implementations.e3nn_lite import TPProblem
from src.benchmark.logging_utils import getLogger

logger = getLogger()

def calculate_minimum_memory_streamed_forward(tpp : TPProblem, batch_size : int) -> dict:
    """
    This represents an absolute minimum amount of memory that could be streamed on an ideal machine without fast memory constraints
    """
    logger.warning("this calcuation does not take into account memory sizes")
    logger.warning("this calcuation does not handle non shared weights yet")
    data_size = {}
    data_size["input 1"] = tpp.irreps_in1.dim * batch_size
    data_size["input 2"] = tpp.irreps_in2.dim * batch_size
    data_size["output"]  = tpp.irreps_out.dim * batch_size
    data_size["weights"] = tpp.weight_numel
    data_size["total"] = sum(data_size.values())
    return data_size

def calculate_minimum_memory_streamed_backward(tpp : TPProblem, batch_size : int) -> dict:
    """
    This represents an absolute minimum amount of memory that could be streamed on an ideal machine without fast memory constraints
    """
    logger.warning("this calcuation does not take into account memory sizes")
    logger.warning("this calcuation does not handle non shared weights yet") 
    data_size = {}
    data_size["input 1"]   = tpp.irreps_in1.dim * batch_size
    data_size["input 1 grad"] = tpp.irreps_in1.dim * batch_size
    data_size["input 2"] = tpp.irreps_in2.dim * batch_size
    data_size["input 2 grad"] = tpp.irreps_in2.dim * batch_size
    data_size["output grad"]  = tpp.irreps_out.dim * batch_size
    data_size["weights"] = tpp.weight_numel
    data_size["weights grad"] = tpp.weight_numel
    data_size["total"] = sum(data_size.values())
    return data_size