import math
from typing import NamedTuple

import numpy as np

from src.benchmark.e3nn_lite_utils import calc_weight_offsets
from src.implementations.e3nn_lite import TPProblem
from src.benchmark.e3nn_lite_utils import Irrep, _MulIr, Irreps, TPProblem, Instruction
from src.implementations.TensorProduct import TensorProduct, GPUInfo
from src.benchmark.logging_utils import getLogger, bcolors 
from jinja2 import Environment, FileSystemLoader

from build.kernel_wrapper import KernelLaunchConfig, JITTPImpl

logger = getLogger()

def raise_helper(msg):
    raise Exception(msg)

def divide(numerator, denominator):
    return numerator // denominator 

def sizeof(dtype):
    if dtype in ["float", "int", "unsigned int"]:
        return 4
    else:
        raise Exception("Provided undefined datatype to sizeof!")

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

class InstructionInfo(NamedTuple):
    """
    This is a class that provides a superset of the information in an Instruction to make templating easier
    """
    # Index Values (Not actually useful)
    in1_index : int
    in2_index : int
    out_index : int
    # Offsets from the base pointer
    in1_offset : int 
    in2_offset : int 
    out_offset : int 
    weight_offset : int
    # Orders 
    in1_l : int 
    in2_l : int 
    out_l : int
    # Multiplicities
    in1_multiplicity : int
    in2_multiplicity : int
    out_multiplicity : int
    # Irrep Length 
    in1_irrep_length : int
    in2_irrep_length : int 
    out_irrep_length : int
    # Tensor Info
    tensor : CGTensor
    path_weight: float
    # Legacy Info (should be accounted for before the templating level)
    connection_mode: str
    has_weight: bool
    path_shape: tuple

def prepare_InstructionInfo_list(problem : TPProblem) -> list[InstructionInfo]:
    """
    This is a convenience funtion that wraps all the info needed at the C++ level into one object
    """
    infolist = []
    L1 = RepData(problem.irreps_in1) 
    L2 = RepData(problem.irreps_in2) 
    L3 = RepData(problem.irreps_out) 
    weight_offsets = calc_weight_offsets(problem)
    assert isinstance(weight_offsets, list)
    assert len(weight_offsets) == len(list(problem.instructions))
    ins : Instruction
    for ins_index, ins in enumerate(problem.instructions): 
        infolist.append(
            InstructionInfo(
                # Irrep Indices 
                in1_index=ins.i_in1,
                in2_index=ins.i_in2,
                out_index=ins.i_out, 
                # Offsets
                in1_offset=L1.offsets[ins.i_in1],
                in2_offset=L2.offsets[ins.i_in2],
                out_offset=L3.offsets[ins.i_out],
                weight_offset=weight_offsets[ins_index],
                # Orders
                in1_l=L1.ls[ins.i_in1],
                in2_l=L2.ls[ins.i_in2],
                out_l=L3.ls[ins.i_out],
                # Multiplicites
                in1_multiplicity=L1.mults[ins.i_in1],
                in2_multiplicity=L2.mults[ins.i_in2],
                out_multiplicity=L3.mults[ins.i_out],
                # Irrep Length 
                in1_irrep_length=L1.irrep_lengths[ins.i_in1],
                in2_irrep_length=L2.irrep_lengths[ins.i_in2],
                out_irrep_length=L3.irrep_lengths[ins.i_out],
                # Tensor Info 
                tensor=CGTensor(L1.ls[ins.i_in1], L2.ls[ins.i_in2], L3.ls[ins.i_out]),
                path_weight=ins.path_weight,
                # Legacy Info 
                connection_mode=ins.connection_mode,
                has_weight=ins.has_weight,
                path_shape=ins.path_shape,
            )
        )
    return infolist

class LoopReorderUVWTP(TensorProduct):
    def __init__(self, config : TPProblem, torch_op : bool = False):
        super().__init__(config, torch_op)

        for ins in config.instructions: # type : Instruction
            assert isinstance(ins, Instruction)
            assert ins.connection_mode == 'uvw'
            assert ins.path_shape[0] <= 32
            assert ins.path_shape[1] <= 32
            assert ins.path_shape[2] <= 32

        irreps_in1 = config.irreps_in1
        irreps_in2 = config.irreps_in2
        irreps_out = config.irreps_out 

        # ==================================================================================

        env = Environment(loader=FileSystemLoader("src/templates"), extensions=['jinja2.ext.do'])
        env.globals['raise'] = raise_helper 
        env.globals['divide'] = divide 
        env.globals['sizeof'] = sizeof
        env.globals['range'] = range
        env.globals['enumerate'] = enumerate 
        env.globals['len'] = len
        env.globals['math.prod'] = math.prod
        main_template = env.get_template("subkernel_per_interaction_multirep.cuh")
        

        # =====================================================================
        # FORWARD MEMORY ANALYSIS 
        forward_thread_blocks_per_SM = 24 
        forward_threads_per_thread_block = 32

        # =====================================================================

        forward_launch_config = KernelLaunchConfig()
        forward_launch_config.num_blocks = GPUInfo.A100_SMS * forward_thread_blocks_per_SM
        forward_launch_config.num_threads = forward_threads_per_thread_block

        # IMPORTANT! 
        smem_gemm_max_n = forward_threads_per_thread_block
        smem_gemm_L3_scratch = smem_gemm_max_n * max(RepData(config.irreps_out).irrep_lengths) # this has space for the largest output size * 32
        smem_gemm_weights_scratch = max(RepData(config.irreps_out).mults) * smem_gemm_max_n

        smem_gemm_info = {
            'n' : smem_gemm_max_n,
            'L3_scratch_elems' : smem_gemm_L3_scratch,
            'weight_scratch_elems' : smem_gemm_weights_scratch,
        }
        logger.debug(smem_gemm_info)
        # END OF IMPORTANT

        forward_launch_config.smem = (
            (irreps_in1.dim + irreps_in2.dim + irreps_out.dim + smem_gemm_L3_scratch + smem_gemm_weights_scratch) 
            * sizeof("float") 
            * forward_launch_config.num_threads // forward_launch_config.warp_size
            ) 

        logger.info(f"Forward pass needs {forward_launch_config.smem} bytes of shared memory.")

        if forward_launch_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {forward_launch_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")
        
        # =====================================================================

        backward_launch_config = KernelLaunchConfig()
        backward_launch_config.num_blocks = GPUInfo.A100_SMS * 1
        backward_launch_config.num_threads = 32
        backward_launch_config.smem = (2 * irreps_in1.dim + 2 * irreps_in2.dim + 2 * + irreps_out.dim)  * sizeof("float") * backward_launch_config.num_threads // backward_launch_config.warp_size 
        logger.info(f"Backward pass needs {backward_launch_config.smem} bytes of shared memory.")

        if backward_launch_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {backward_launch_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

        # =====================================================================     

        self.forward_config = forward_launch_config
        self.backward_config = backward_launch_config 

        # =====================================================================
        kernel_text = main_template.render(
            problem=config,
            InstructionInfoList = prepare_InstructionInfo_list(config),
            smem_gemm_info=smem_gemm_info,
            forward_config=forward_launch_config,
            backward_config=backward_launch_config
        )

        self.jit_kernel = kernel_text
        
        logger.debug(kernel_text)

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")

    @classmethod
    def name(cls):
        return cls.__name__
