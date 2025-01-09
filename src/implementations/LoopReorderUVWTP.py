import math, logging, pathlib
from typing import NamedTuple, Literal, get_args

import numpy as np

from src.benchmark.e3nn_lite_utils import calc_weight_offsets
from src.implementations.e3nn_lite import TPProblem
from src.benchmark.e3nn_lite_utils import Irrep, _MulIr, Irreps, TPProblem, Instruction
from src.implementations.TensorProduct import TensorProduct
from src.benchmark.logging_utils import getLogger, bcolors 
from jinja2 import Environment, FileSystemLoader

from build.kernel_wrapper import KernelLaunchConfig, JITTPImpl, DeviceProp

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
    # Weight Sub Partitioning Info
    weight_in1_extent : int
    weight_in2_extent : int 
    weight_out_extent : int 
    weight_in1_offset : int
    weight_in2_offset : int 
    weight_out_offset : int 

Dimension = Literal['in1', 'in2', 'out']

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
                # Weight Sub Partitioning Info
                weight_in1_extent=L1.mults[ins.i_in1],
                weight_in2_extent=L2.mults[ins.i_in2],
                weight_out_extent=L3.mults[ins.i_out],
                weight_in1_offset=0, 
                weight_in2_offset=0, 
                weight_out_offset=0, 
            )
        )
    return infolist

def partition_InstructionInfo_list_by_max_size_along_dimension(input_II_list : list[InstructionInfo], max_size : int, dimension : Dimension) -> list[InstructionInfo]: 
    assert dimension in get_args(Dimension)
    output_II_list = []
    while input_II_list:
        II = input_II_list.pop()
        extent = getattr(II,f"{dimension}_multiplicity")
        assert isinstance(extent, int)

        if extent > max_size:
            # hunk is the max_sized bit 
            # rest is the rest of it   

            irrep_offsets : dict[Dimension, int]= {
                'in1' : II.in1_multiplicity, 
                'in2' : II.in2_multiplicity, 
                'out' : II.out_multiplicity, 
            }
            hunk_irrep_offsets = irrep_offsets.copy()
            rest_irrep_offsets = irrep_offsets.copy()
            
            rest_irrep_offsets[dimension] += max_size

            weight_offsets : dict[Dimension, int]= {
                'in1' : II.weight_in1_offset,
                'in2' : II.weight_in2_offset,
                'out' : II.weight_out_offset, 
            }
            hunk_weight_offsets = irrep_offsets.copy()
            rest_weight_offsets = irrep_offsets.copy() 

            rest_weight_offsets[dimension] += max_size

            multiplicities : dict[Dimension, int] = {
                'in1' : II.in1_multiplicity,
                'in2' : II.in2_multiplicity, 
                'out' : II.out_multiplicity, 
            }
            hunk_multiplicities = multiplicities.copy()
            rest_multiplicities = multiplicities.copy()

            hunk_multiplicities[dimension]  = max_size
            rest_multiplicities[dimension] -= max_size 

            rest_II = InstructionInfo(
                # Irrep Indices 
                in1_index=II.in1_index, # This won't acutally be accurate with the partition, but it will correspond to the original blocks
                in2_index=II.in2_index,
                out_index=II.out_index, 
                # Offsets
                in1_offset=rest_irrep_offsets['in1'],
                in2_offset=rest_irrep_offsets['in2'],
                out_offset=rest_irrep_offsets['out'], 
                weight_offset=II.weight_offset,
                # Orders
                in1_l=II.in1_l,
                in2_l=II.in2_l,
                out_l=II.out_l,
                # Multiplicites
                in1_multiplicity=rest_multiplicities['in1'],
                in2_multiplicity=rest_multiplicities['in2'],
                out_multiplicity=rest_multiplicities['out'],
                # Irrep Length 
                in1_irrep_length=II.in1_irrep_length,
                in2_irrep_length=II.in2_irrep_length,
                out_irrep_length=II.out_irrep_length,
                # Tensor Info 
                tensor=II.tensor,
                path_weight=II.path_weight,
                # Legacy Info 
                connection_mode=II.connection_mode,
                has_weight=II.has_weight,
                path_shape=(rest_multiplicities['in1'], rest_multiplicities['in2'], rest_multiplicities['out']),
                # Weight Sub Partitioning Info
                weight_in1_extent=II.weight_in1_extent,
                weight_in2_extent=II.weight_in2_extent,
                weight_out_extent=II.weight_out_extent,
                weight_in1_offset=rest_weight_offsets['in1'], 
                weight_in2_offset=rest_weight_offsets['in2'], 
                weight_out_offset=rest_weight_offsets['out'], 
            )

            hunk_II = InstructionInfo(
                # Irrep Indices 
                in1_index=II.in1_index, # This won't acutally be accurate with the partition, but it will correspond to the original blocks
                in2_index=II.in2_index,
                out_index=II.out_index, 
                # Offsets
                in1_offset=hunk_irrep_offsets['in1'],
                in2_offset=hunk_irrep_offsets['in2'],
                out_offset=hunk_irrep_offsets['out'], 
                weight_offset=II.weight_offset,
                # Orders
                in1_l=II.in1_l,
                in2_l=II.in2_l,
                out_l=II.out_l,
                # Multiplicites
                in1_multiplicity=hunk_multiplicities['in1'],
                in2_multiplicity=hunk_multiplicities['in2'],
                out_multiplicity=hunk_multiplicities['out'],
                # Irrep Length 
                in1_irrep_length=II.in1_irrep_length,
                in2_irrep_length=II.in2_irrep_length,
                out_irrep_length=II.out_irrep_length,
                # Tensor Info 
                tensor=II.tensor,
                path_weight=II.path_weight,
                # Legacy Info 
                connection_mode=II.connection_mode,
                has_weight=II.has_weight,
                path_shape=(hunk_multiplicities['in1'], hunk_multiplicities['in2'], hunk_multiplicities['out']),
                # Weight Sub Partitioning Info
                weight_in1_extent=II.weight_in1_extent,
                weight_in2_extent=II.weight_in2_extent,
                weight_out_extent=II.weight_out_extent,
                weight_in1_offset=hunk_weight_offsets['in1'], 
                weight_in2_offset=hunk_weight_offsets['in2'], 
                weight_out_offset=hunk_weight_offsets['out'], 
            )
            output_II_list.append(hunk_II)
            input_II_list.append(rest_II)   
        else: 
            output_II_list.append(II)      
    return output_II_list



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
        env.globals['product'] = math.prod
        main_template = env.get_template("loop_reorder_uvw_cutlass_tp.cuh")

        subkernel_template = env.get_template("loop_reorder_subkernels.cuh")
        
        # ===========================================================================
        # Memory Analyis for Smem Allocation
        dp = DeviceProp(0)

        logger.debug(msg=f"available smem = {dp.maxSharedMemPerBlock}")

        InstructionInfoList = prepare_InstructionInfo_list(config)

        logger.debug(msg=f"original II list{InstructionInfoList}")

        InstructionInfoList = partition_InstructionInfo_list_by_max_size_along_dimension(InstructionInfoList, 8, 'in1')
        InstructionInfoList = partition_InstructionInfo_list_by_max_size_along_dimension(InstructionInfoList, 8, 'in2')
        InstructionInfoList = partition_InstructionInfo_list_by_max_size_along_dimension(InstructionInfoList, 8, 'out')

        logger.debug(msg=f"partitioned II list{InstructionInfoList}")

        max_in1_instruction_size = max([II.in1_irrep_length * II.in1_multiplicity for II in InstructionInfoList])
        max_in2_instruction_size = max([II.in2_irrep_length * II.in2_multiplicity for II in InstructionInfoList])
        max_out_instruction_size = max([II.out_irrep_length * II.out_multiplicity for II in InstructionInfoList])

        max_weight_size = max([math.prod(II.path_shape) for II in InstructionInfoList])

        # =====================================================================
        # FORWARD MEMORY ANALYSIS 
        forward_thread_blocks_per_SM = 1 

        # =====================================================================

        forward_launch_config = KernelLaunchConfig()
        forward_launch_config.num_blocks = dp.multiprocessorCount * forward_thread_blocks_per_SM

         # IMPORTANT! 
        forward_smem_gemm_max_n = dp.warpsize
        forward_smem_gemm_L3_scratch = forward_smem_gemm_max_n * max([II.out_irrep_length for II in InstructionInfoList]) # this has space for the largest output size * 32
        foward_smem_gemm_weights_scratch = max(RepData(config.irreps_out).mults) * forward_smem_gemm_max_n


        forward_smem_gemm_info = {
            'n' : forward_smem_gemm_max_n,
            'L3_scratch_elems' : forward_smem_gemm_L3_scratch,
            'weight_scratch_elems' : foward_smem_gemm_weights_scratch,

        }

        logger.debug(f"{forward_smem_gemm_info=}")
        # END OF IMPORTANT

        forward_smem_per_warp = {}
        forward_smem_common = {}

        forward_smem_per_warp['in1'] =       max_in1_instruction_size * sizeof('float')
        forward_smem_per_warp['in2'] =       max_in2_instruction_size * sizeof('float')
        forward_smem_per_warp['out'] =       max_out_instruction_size * sizeof('float')
        forward_smem_per_warp['gemm_L1L2'] = forward_smem_gemm_L3_scratch * sizeof('float')
        forward_smem_per_warp['gemm_weights'] = (foward_smem_gemm_weights_scratch * sizeof('float'))


        forward_smem_common['weights'] = max_weight_size * sizeof('float')
    
        forward_smem_per_warp_total = sum(forward_smem_per_warp.values())
        forwad_smem_common_total = sum(forward_smem_common.values())
        
        logger.debug(f"{forward_smem_per_warp=}")
        logger.debug(f"{forward_smem_common=}")                

        forward_num_warps_that_fit = max(dp.maxSharedMemPerBlock - forwad_smem_common_total, 0)  // forward_smem_per_warp_total
        assert (forward_num_warps_that_fit > 0)
        forward_num_warps_sane = 16 

        forward_num_warps = min(forward_num_warps_that_fit, forward_num_warps_sane)
        logger.debug(f"{forward_num_warps=}") 

        forward_launch_config.num_threads = (dp.warpsize * forward_num_warps) 
        forward_launch_config.smem = (forward_smem_per_warp_total * forward_num_warps) + forwad_smem_common_total 

        logger.debug(f"Forward pass needs {forward_launch_config.smem} bytes of shared memory.")

        if forward_launch_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {forward_launch_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")
        
        # =====================================================================

        backward_launch_config = KernelLaunchConfig()
        backward_launch_config.num_blocks = dp.multiprocessorCount * 1


        backward_smem_gemm_max_n = dp.warpsize
        backward_smem_gemm_L1L2_scratch = backward_smem_gemm_max_n * max(RepData(config.irreps_out).irrep_lengths) # this has space for the largest output size * 32
        backward_smem_gemm_weights_scratch = backward_smem_gemm_max_n * max(RepData(config.irreps_out).mults) 

        backward_smem_gemm_info = {
            'n' : backward_smem_gemm_max_n,
            'L1L2_scratch_elems' : backward_smem_gemm_L1L2_scratch,
            'weight_scratch_elems' : backward_smem_gemm_weights_scratch,
        }

        backward_smem_per_warp = {}
        backward_smem_common = {}

        backward_smem_per_warp['in1']          = max_in1_instruction_size * sizeof('float')
        backward_smem_per_warp['in1_grad']     = max_in1_instruction_size * sizeof('float')
        backward_smem_per_warp['in2']          = max_in2_instruction_size * sizeof('float')
        backward_smem_per_warp['in2_grad']     = max_in2_instruction_size * sizeof('float')
        backward_smem_per_warp['out_grad']     = max_out_instruction_size * sizeof('float')
        backward_smem_per_warp['gemm_L1L2']    = backward_smem_gemm_L1L2_scratch * sizeof('float')
        backward_smem_per_warp['gemm_weights'] = backward_smem_gemm_weights_scratch * sizeof('float')

        backward_smem_common['weights'] = max_weight_size * sizeof('float')
        backward_smem_common['weights_grad'] = max_weight_size * sizeof('float')
        
        backward_smem_per_warp_total = sum(backward_smem_per_warp.values())
        backward_smem_common_total = sum(backward_smem_common.values())


        logger.debug(msg=f"{backward_smem_per_warp=}")
        logger.debug(msg=f"{backward_smem_common=}")
        

        backward_num_warps_that_fit = max(dp.maxSharedMemPerBlock - backward_smem_common_total, 0) // backward_smem_per_warp_total
        assert(backward_num_warps_that_fit > 0)
        backward_num_warps_sane_limit = 16        

        backward_num_warps = min(backward_num_warps_that_fit, backward_num_warps_sane_limit)
        logger.debug(msg=f"{backward_num_warps=}")

        backward_launch_config.warp_size = dp.warpsize
        backward_launch_config.num_threads = backward_launch_config.warp_size * backward_num_warps
        backward_launch_config.smem = (backward_smem_per_warp_total * backward_num_warps) + backward_smem_common_total

        logger.debug(f"Backward pass needs {backward_launch_config.smem} bytes of shared memory.")
        
        if backward_launch_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {backward_launch_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")

        # =====================================================================     

        self.forward_config = forward_launch_config
        self.backward_config = backward_launch_config 

        # =====================================================================
        kernel_text = main_template.render(
            problem=config,
            L1=RepData(config.irreps_in1), 
            L2=RepData(config.irreps_in2), 
            L3=RepData(config.irreps_out),
            InstructionInfoList = InstructionInfoList,
            max_in1_instruction_size = max_in1_instruction_size,
            max_in2_instruction_size = max_in2_instruction_size,
            max_out_instruction_size = max_out_instruction_size,
            max_weight_size = max_weight_size,
            forward_smem_gemm_info=forward_smem_gemm_info,
            backward_smem_gemm_info=backward_smem_gemm_info,
            forward_config=forward_launch_config,
            backward_config=backward_launch_config
        )

        for i, II in enumerate(InstructionInfoList):
            kernel_text += subkernel_template.render(
                kernel_ID = i,
                II = II,
            )

        self.jit_kernel = kernel_text
        
        def add_fixed_width_line_numbers(text):
            lines = text.split('\n')
            numbered_lines = [f"{i + 1:03}: {line}" for i, line in enumerate(lines)]
            return '\n'.join(numbered_lines)
            
        if logger.isEnabledFor(logging.DEBUG):
            logs_path = pathlib.Path('logs/')
            logs_path.mkdir(parents=True, exist_ok=True)
            with open(logs_path / "kernel_text.txt", 'w') as f:
                f.write(kernel_text)

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")

    @classmethod
    def name(cls):
        return cls.__name__
