import logging, pathlib
import numpy as np

from src.benchmark.e3nn_lite_utils import calc_weight_offsets
from src.benchmark.e3nn_lite_utils import CGTensor, RepData
from src.implementations.e3nn_lite import TPProblem, Instruction
from src.implementations.TensorProduct import TensorProduct 
from src.implementations.InstructionInfo import prepare_InstructionInfo_list, partition_InstructionInfo_list_by_max_size_along_dimension, pretty_format_InstructionInfoList
from src.benchmark.logging_utils import getLogger, bcolors 
from src.templates.jinja_utils import get_jinja_environment, sizeof
from jinja2 import Environment, FileSystemLoader

from build.kernel_wrapper import KernelLaunchConfig, JITTPImpl, DeviceProp

logger = getLogger()

class MultiplicityOuterProductTP(TensorProduct):
    def __init__(self, config : TPProblem, torch_op : bool = False):
        super().__init__(config, torch_op)

        for ins in config.instructions: # type : Instruction
            assert isinstance(ins, Instruction)
            assert ins.connection_mode == 'uvw'
            # assert ins.path_shape[0] <= 32
            # assert ins.path_shape[1] <= 32
            # assert ins.path_shape[2] <= 32

        irreps_in1 = config.irreps_in1
        irreps_in2 = config.irreps_in2
        irreps_out = config.irreps_out 

        # ==================================================================================

        env = get_jinja_environment()
        main_template = env.get_template("mop_tp.cuh")
        # forward_subkernel_template = env.get_template("subkernel_forward_thread.cu.jinja2")
        # backward_subkernel_template = env.get_template("subkernel_backward_thread.cu.jinja2")
        
        # =====================================================================
        # FORWARD MEMORY ANALYSIS 
        forward_thread_blocks_per_SM = 2  
        
        # =====================================================================
        dp = DeviceProp(0)

        InstructionInfoList = prepare_InstructionInfo_list(config)

        InstructionInfoList = partition_InstructionInfo_list_by_max_size_along_dimension(InstructionInfoList, 32, 'in1')
        InstructionInfoList = partition_InstructionInfo_list_by_max_size_along_dimension(InstructionInfoList, 32, 'in2')
        InstructionInfoList = partition_InstructionInfo_list_by_max_size_along_dimension(InstructionInfoList, 32, 'out')

        logger.debug(msg="Initial II list")
        logger.debug(msg=pretty_format_InstructionInfoList(InstructionInfoList))

        max_in1_instruction_size = max([II.in1_irrep_length * II.in1_multiplicity for II in InstructionInfoList])
        max_in2_instruction_size = max([II.in2_irrep_length * II.in2_multiplicity for II in InstructionInfoList])
        max_out_instruction_size = max([II.out_irrep_length * II.out_multiplicity for II in InstructionInfoList])

        # =====================================================================

        forward_launch_config = KernelLaunchConfig()
        forward_launch_config.num_blocks = dp.multiprocessorCount * forward_thread_blocks_per_SM

        forward_smem_per_warp = {}
        forward_smem_per_warp['in1'] = (irreps_in1.dim * sizeof('float'))
        forward_smem_per_warp['in2'] = (irreps_in2.dim * sizeof('float'))
        forward_smem_per_warp['out'] = (irreps_out.dim * sizeof('float'))

        # IMPORTANT! 
        forward_smem_gemm_max_n = dp.warpsize
        forward_smem_gemm_L3_scratch = forward_smem_gemm_max_n * max(RepData(config.irreps_out).irrep_lengths) # this has space for the largest output size * 32
        foward_smem_gemm_weights_scratch = max(RepData(config.irreps_out).mults) * forward_smem_gemm_max_n

        forward_smem_gemm_info = {
            'n' : forward_smem_gemm_max_n,
            'L3_scratch_elems' : forward_smem_gemm_L3_scratch,
            'weight_scratch_elems' : foward_smem_gemm_weights_scratch,
        }

        logger.debug(f"{forward_smem_gemm_info=}")
        # END OF IMPORTANT

        forward_smem_per_warp['gemm_L3'] = (forward_smem_gemm_L3_scratch * sizeof('float'))
        forward_smem_per_warp['gemm_weights'] = (foward_smem_gemm_weights_scratch * sizeof('float'))

        logger.debug(f"{forward_smem_per_warp=}")

        forward_smem_per_warp_total = sum(forward_smem_per_warp.values())

        forward_num_warps_that_fit = dp.maxSharedMemPerBlock // forward_smem_per_warp_total
        forward_num_warps_sane = 16 

        forward_num_warps = min(forward_num_warps_that_fit, forward_num_warps_sane)

        forward_launch_config.num_threads = dp.warpsize * forward_num_warps
        forward_launch_config.smem = forward_smem_per_warp_total * forward_num_warps 

        logger.debug(f"Forward pass needs {forward_launch_config.smem} bytes of shared memory.")

        
        
        # =====================================================================

        backward_launch_config = KernelLaunchConfig()
        backward_launch_config.num_blocks = dp.multiprocessorCount * 1 


        backward_smem_gemm_max_n = dp.warpsize
        backward_smem_gemm_L1L2_scratch = backward_smem_gemm_max_n * max(RepData(config.irreps_out).irrep_lengths) # this has space for the largest output size * 32
        backward_smem_gemm_weights_scratch = backward_smem_gemm_max_n * max(RepData(config.irreps_out).mults) 

        backward_smem_per_warp = {}
        backward_smem_per_warp['in1']          = (irreps_in1.dim * sizeof('float'))
        backward_smem_per_warp['in1_grad']     = (irreps_in1.dim * sizeof('float'))
        backward_smem_per_warp['in2']          = (irreps_in2.dim * sizeof('float'))
        backward_smem_per_warp['in2_grad']     = (irreps_in2.dim * sizeof('float'))
        backward_smem_per_warp['out_grad']     = (irreps_out.dim * sizeof('float'))
        backward_smem_per_warp['gemm_L1L2']    = (backward_smem_gemm_L1L2_scratch * sizeof('float'))
        backward_smem_per_warp['gemm_weights'] = (backward_smem_gemm_weights_scratch * sizeof('float'))
        
        backward_smem_gemm_info = {
            'n' : backward_smem_gemm_max_n,
            'L1L2_scratch_elems' : backward_smem_gemm_L1L2_scratch,
            'weight_scratch_elems' : backward_smem_gemm_weights_scratch,
        }

        logger.debug(msg=f"{backward_smem_per_warp=}")

        backward_smem_per_warp_total = sum(backward_smem_per_warp.values())

        logger.debug(msg=f"{backward_smem_per_warp_total=}")

        backward_num_warps_that_fit = dp.maxSharedMemPerBlock // backward_smem_per_warp_total
        backward_num_warps_sane_limit = 16

        backward_num_warps = min(backward_num_warps_that_fit, backward_num_warps_sane_limit)
        logger.info(msg=f"{backward_num_warps=}")

        backward_launch_config.warp_size = dp.warpsize
        backward_launch_config.num_threads = backward_launch_config.warp_size * backward_num_warps
        backward_launch_config.smem = backward_smem_per_warp_total * backward_num_warps

        logger.debug(f"Backward pass needs {backward_launch_config.smem} bytes of shared memory.")

        if backward_launch_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {backward_launch_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")

        # =====================================================================     

        self.forward_config = forward_launch_config
        self.backward_config = backward_launch_config 

        # =====================================================================
        # weights_offsets
        weight_offsets = calc_weight_offsets(config)
        assert isinstance(weight_offsets, list)
        assert len(weight_offsets) == len(list(config.instructions))

        # =====================================================================
        # tranform "e3nn instructions" into "interactions"
        instructions : list[Instruction] = config.instructions 
        interactions = []
        for ins in instructions:
            u = ins.i_in1
            v = ins.i_in2
            w = ins.i_out
            interaction = (u, v, w, CGTensor(irreps_in1[u].ir.l, irreps_in2[v].ir.l, irreps_out[w].ir.l))
            interactions.append(interaction)


        assert len(interactions) != 0

        # =====================================================================
        kernel_text = main_template.render(
            L1=RepData(config.irreps_in1), 
            L2=RepData(config.irreps_in2), 
            L3=RepData(config.irreps_out),
            InstructionInfoList=InstructionInfoList,
            max_in1_instruction_size=max_in1_instruction_size,
            max_in2_instruction_size=max_in2_instruction_size,
            max_out_instruction_size=max_out_instruction_size,
            weight_numel=config.weight_numel,
            weight_offsets=weight_offsets,
            instructions=instructions,
            interactions=interactions,
            forward_smem_gemm_info=forward_smem_gemm_info,
            backward_smem_gemm_info=backward_smem_gemm_info,
            forward_config=forward_launch_config,
            backward_config=backward_launch_config
        )   

        self.jit_kernel = kernel_text
        
        if logger.isEnabledFor(logging.DEBUG):
            logs_path = pathlib.Path('logs/')
            logs_path.mkdir(parents=True, exist_ok=True)
            with open(logs_path / "kernel_text.txt", 'w') as f:
                f.write(kernel_text)

        logger.debug("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.debug("Kernel compiled!")


    @classmethod
    def name(cls):
        return cls.__name__