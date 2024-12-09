import numpy as np
from build.kernel_wrapper import *
from src.templates.jinja_utils import *
from src.implementations.ComputationSchedule import ComputationSchedule 

from src.implementations.TensorProduct import TensorProduct 
from src.benchmark.logging_utils import getLogger, bcolors
from src.benchmark.e3nn_lite_utils import count_cg_non_zero
logger = getLogger()

class LoopUnrollTP(TensorProduct):
    def __init__(self, config, torch_op=False):
        super().__init__(config, torch_op=torch_op)
        L1, L2, L3 = self.L1, self.L2, self.L3 

        for (mul, ir) in L2:
            assert(mul == 1)

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_batch.cuh")
        env.globals['enumerate'] = enumerate 

        dp = DeviceProp(0)

        forward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=6,
                block_count=dp.multiprocessorCount * 3,
                direction = "forward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype)

        backward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=4,
                block_count=dp.multiprocessorCount * 4,
                direction = "backward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype)

        self.jit_kernel = template.render(
            forward_schedule=forward_schedule,
            backward_schedule=backward_schedule)

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel,
                forward_schedule.launch_config,
                backward_schedule.launch_config)
        logger.info("Kernel compiled!")

        logger.info(f"CUDA Kernel File Size: {len(self.jit_kernel) // 1000} KB")

    @staticmethod
    def name():
        return "LoopUnrollTP"
 
    def calculate_flops_forward(self, batch_size : int) -> dict:
        tpp = self.config
        flop_count = {'CG_decomposition': 0, 'linear_combination': 0, 'outer_products': 0}
        for ins in tpp.instructions: 
            l1, l2, l3 = tpp.irreps_in1[ins.i_in1].ir.l, tpp.irreps_in2[ins.i_in2].ir.l, tpp.irreps_out[ins.i_out].ir.l
            flop_count["CG_decomposition"] += count_cg_non_zero(l1, l2, l3) * (ins.path_shape[0] * ins.path_shape[1])
            flop_count["linear_combination"] += (2 * l3 + 1) * np.prod(ins.path_shape) if ins.has_weight else 0

        flop_count["CG_decomposition"] *= 3 * batch_size
        flop_count["linear_combination"] *= batch_size    # Weights do not require FMA here
        flop_count["total"] = sum(flop_count.values())
        return flop_count

    def calculate_flops_backward(self, batch_size : int) -> dict:
        tpp = self.config
        flop_count = {'backward': 0} 
        for ins in tpp.instructions: 
            l1, l2, l3 = tpp.irreps_in1[ins.i_in1].ir.l, tpp.irreps_in2[ins.i_in2].ir.l, tpp.irreps_out[ins.i_out].ir.l
            flop_count["backward"] += count_cg_non_zero(l1, l2, l3) * (ins.path_shape[0] * ins.path_shape[1])

        flop_count["backward"] *= 9 * batch_size
        flop_count["total"] = sum(flop_count.values())
        return flop_count
