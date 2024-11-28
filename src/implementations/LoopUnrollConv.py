from src.implementations.Convolution import *
from src.implementations.TensorProduct import TensorProduct
from src.implementations.ComputationSchedule import ComputationSchedule 
from src.templates.jinja_utils import *
from build.kernel_wrapper import *

class LoopUnrollConv(Convolution):
    def __init__(self, config):
        super().__init__(config)
        L1, L2, L3 = self.L1, self.L2, self.L3 

        for (mul, ir) in L2:
            assert(mul == 1)

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv.cuh")
        env.globals['enumerate'] = enumerate 

        dp = DeviceProp(0)

        forward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=6,
                block_count=dp.multiprocessorCount * 3,
                direction = "forward",
                irrep_dtype = np.float32,
                weight_dtype = np.float32)

        backward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=4,
                block_count=dp.multiprocessorCount * 4,
                direction = "backward",
                irrep_dtype = np.float32,
                weight_dtype = np.float32)

        for sched in [forward_schedule, backward_schedule]:
            for segment in sched.segments:
                for map in segment.maps:
                    for key in map.storeback_procedure:
                        map.storeback_procedure[key] = "atomic_accumulate"

        self.jit_kernel = template.render(
            forward_schedule=forward_schedule,
            backward_schedule=backward_schedule)

        logger.info("Starting NVRTC")
        self.internal = JITConvImpl(self.jit_kernel,
                forward_schedule.launch_config, 
                backward_schedule.launch_config)
        logger.info("Kernel compiled!")


    @staticmethod
    def name():
        return "LoopUnrollConv"