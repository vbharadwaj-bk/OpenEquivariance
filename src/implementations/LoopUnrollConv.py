from src.implementations.Convolution import *
from src.implementations.ComputationSchedule import ComputationSchedule 
from src.templates.jinja_utils import *
from build.kernel_wrapper import *

class LoopUnrollConv(Convolution):
    def __init__(self, config, 
            idx_dtype=np.int64, 
            torch_op=False,
            deterministic=False):
        super().__init__(config, idx_dtype, torch_op, deterministic)
        L1, L2, L3 = self.L1, self.L2, self.L3 

        for (mul, ir) in L2:
            assert(mul == 1)

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv_atomic.cuh")
        env.globals['enumerate'] = enumerate 

        dp = DeviceProp(0)

        schedule_type = 3
        if deterministic:
            schedule_type = 3
            template = env.get_template("loop_unroll_conv_det.cuh")

        forward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=6,
                block_count=dp.multiprocessorCount,
                direction = "forward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype,
                schedule_type=schedule_type)

        backward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=4,
                block_count=dp.multiprocessorCount * 4,
                direction = "backward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype,
                schedule_type=schedule_type)

        if not deterministic:
            for segment in forward_schedule.segments:
                for key in segment.L3Map.storeback_procedure:
                    segment.L3Map.storeback_procedure[key] = "atomic_accumulate"

            for segment in backward_schedule.segments:
                for key in segment.L1Map.storeback_procedure:
                    segment.L1Map.storeback_procedure[key] = "atomic_accumulate"

        idx_type_map = {np.int32: "int", np.int64: "long"}

        self.jit_kernel = template.render(
            forward_schedule=forward_schedule,
            backward_schedule=backward_schedule,
            idx_type=idx_type_map[idx_dtype])

        logger.info("Starting NVRTC")
        self.internal = JITConvImpl(self.jit_kernel,
                forward_schedule.launch_config, 
                backward_schedule.launch_config)
        logger.info("Kernel compiled!")

        if self.torch_op:
            self.setup_torch_module()

        if deterministic:
            workspace_size = max(
                (forward_schedule.L3.dim * np.dtype(config.irrep_dtype).itemsize + 4) * forward_schedule.total_warps,
                (backward_schedule.L1.dim * np.dtype(config.irrep_dtype).itemsize + 4) * backward_schedule.total_warps)
            self.allocate_workspace(workspace_size)


    @staticmethod
    def name():
        return "LoopUnrollConv"

class LoopUnrollConvDeterministic(LoopUnrollConv):
    def __init__(self, config, 
            idx_dtype=np.int64, 
            torch_op=False):
        super().__init__(config, idx_dtype, torch_op, deterministic=True)

    @staticmethod
    def name():
        return "LoopUnrollConvDeterministic"

class LoopUnrollConvAtomic(LoopUnrollConv):
    def __init__(self, config, 
            idx_dtype=np.int64, 
            torch_op=False):
        super().__init__(config, idx_dtype, torch_op, deterministic=False)

    @staticmethod
    def name():
        return "LoopUnrollConvAtomic"