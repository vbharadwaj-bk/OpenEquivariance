import numpy as np

from src.benchmark.e3nn_lite_utils import calc_weight_offsets
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

class MultiplicityOuterProductTP(TensorProduct):
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
        main_template = env.get_template("mop_tp.cuh")
        # forward_subkernel_template = env.get_template("subkernel_forward_thread.cu.jinja2")
        # backward_subkernel_template = env.get_template("subkernel_backward_thread.cu.jinja2")
        
        # =====================================================================
        # Updated to work with TensorProductProblem
    
        class RepData:
            def __init__(self, irreps : Irreps):
                assert isinstance(irreps, Irreps)
                self.rep_len = irreps.dim
                self.irrep_lengths = [mul_irrep.ir.dim for mul_irrep in irreps]
                self.mults = [mul_irrep.mul for mul_irrep in irreps]

                offset = 0
                self.offsets = []
                for mul_irrep in irreps: 
                    self.offsets.append(offset)
                    offset += mul_irrep.dim

        # =====================================================================
        # Strictly Copied from Loop Unroll TP

        class CGTensor:
            def __init__(self, l1, l2, l3):
                tensor = load_cg_tensor(l1, l2, l3)
                coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
                values = [str(float.hex(float(val))) + "f" for val in float_values]

                self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
                self.nnz = len(values)

        # =====================================================================
        # FORWARD MEMORY ANALYSIS 
        forward_thread_blocks_per_SM = 2  
        

        # =====================================================================
        dp = DeviceProp(0)

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
        forward_num_warps_sane = 8 

        forward_num_warps = min(forward_num_warps_that_fit, forward_num_warps_sane)

        forward_launch_config.num_threads = dp.warpsize * forward_num_warps
        forward_launch_config.smem = forward_smem_per_warp_total * forward_num_warps 

        logger.debug(f"Forward pass needs {forward_launch_config.smem} bytes of shared memory.")

        if forward_launch_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {forward_launch_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")
        
        # =====================================================================

        backward_launch_config = KernelLaunchConfig()
        backward_launch_config.num_blocks = dp.multiprocessorCount * 2

        backward_smem_gemm_max_n = dp.warpsize
        backward_smem_gemm_L1L2_scratch = backward_smem_gemm_max_n * max(RepData(config.irreps_out).irrep_lengths) # this has space for the largest output size * 32
        backward_smem_gemm_weights_scratch = max(RepData(config.irreps_out).mults) * backward_smem_gemm_max_n

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
        backward_num_warps_sane_limit = 8

        backward_num_warps = min(backward_num_warps_that_fit, backward_num_warps_sane_limit)
        logger.info(msg=f"{backward_num_warps=}")

        backward_launch_config.num_threads = backward_launch_config.warp_size * backward_num_warps
        backward_launch_config.smem = backward_smem_per_warp_total * backward_num_warps

        logger.debug(f"Backward pass needs {backward_launch_config.smem} bytes of shared memory.")

        if backward_launch_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {backward_launch_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")

        # =====================================================================     

        self.forward_config = forward_launch_config
        self.backward_config = backward_launch_config 
        load_cg_tensor = self.load_cg_tensor

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
            weight_numel=config.weight_numel,
            weight_offsets=weight_offsets,
            instructions=instructions,
            interactions=interactions,
            forward_smem_gemm_info=forward_smem_gemm_info,
            backward_smem_gemm_info=backward_smem_gemm_info
            forward_config=forward_launch_config,
            backward_config=backward_launch_config
        )   

        self.jit_kernel = kernel_text
        
        def add_fixed_width_line_numbers(text):
            lines = text.split('\n')
            numbered_lines = [f"{i + 1:03}: {line}" for i, line in enumerate(lines)]
            return '\n'.join(numbered_lines)
            
        logger.debug(add_fixed_width_line_numbers(kernel_text))

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")


    @classmethod
    def name(cls):
        return cls.__name__