from fast_tp.implementations.convolution.Convolution import *
from fast_tp.implementations.ComputationSchedule import ComputationSchedule
from fast_tp.implementations.LoopUnrollTP import *
from fast_tp.templates.jinja_utils import *
from fast_tp.extlib.kernel_wrapper import *

class LoopUnrollConv(Convolution):
    def __init__(self, config, idx_dtype=np.int64, 
            torch_op=False, deterministic=False):
        super().__init__(config, idx_dtype, torch_op, deterministic)
        L1, L2, L3 = self.L1, self.L2, self.L3 

        for (mul, ir) in L2:
            assert(mul == 1)

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv_atomic.cuh")
        env.globals['enumerate'] = enumerate 

        dp = DeviceProp(0)

        forward_schedule_type = 3
        backward_schedule_type = 2
        if deterministic:
            backward_schedule_type = 3
            template = env.get_template("loop_unroll_conv_det.cuh")

        forward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock // 4 * 3, warps_per_block=6,
                block_count=dp.multiprocessorCount,
                direction = "forward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype,
                schedule_type=forward_schedule_type)

        backward_schedule = ComputationSchedule(self.config, 
                smem_limit=dp.maxSharedMemPerBlock, warps_per_block=6,
                block_count=dp.multiprocessorCount * 2,
                direction = "backward",
                irrep_dtype = config.irrep_dtype,
                weight_dtype = config.weight_dtype,
                schedule_type=backward_schedule_type)

        if not deterministic:
            for segment in forward_schedule.segments:
                for key in segment.L3Map.storeback_procedure:
                    segment.L3Map.storeback_procedure[key] = "atomic_accumulate"

            for segment in backward_schedule.segments:
                for key in segment.L1Map.storeback_procedure:
                    segment.L1Map.storeback_procedure[key] = "atomic_accumulate"

        idx_type_map = {np.int32: "int", np.int64: "long"}

        if self.torch_op:
            self.setup_torch_module()

        self.forward_workspace_offset = None
        self.backward_workspace_offset = None

        if deterministic:
            destination_index_bytes = 32 # Add extra to account for padding
            workspace_size = max(
                (forward_schedule.L3.dim * np.dtype(config.irrep_dtype).itemsize + destination_index_bytes) * forward_schedule.total_warps,
                (backward_schedule.L1.dim * np.dtype(config.irrep_dtype).itemsize + destination_index_bytes) * backward_schedule.total_warps)
            self.allocate_workspace(workspace_size)

            self.forward_workspace_offset = forward_schedule.L3.dim * np.dtype(config.irrep_dtype).itemsize * forward_schedule.total_warps
            self.backward_workspace_offset = backward_schedule.L1.dim * np.dtype(config.irrep_dtype).itemsize * backward_schedule.total_warps

            self.forward_workspace_offset = (self.forward_workspace_offset + 7) // 8 * 8
            self.backward_workspace_offset = (self.backward_workspace_offset + 7) // 8 * 8


        self.jit_kernel = template.render(
            forward_schedule=forward_schedule,
            backward_schedule=backward_schedule,
            idx_type=idx_type_map[idx_dtype],
            forward_workspace_offset=self.forward_workspace_offset,
            backward_workspace_offset=self.backward_workspace_offset)

        logger.info("Starting NVRTC")
        self.internal = JITConvImpl(self.jit_kernel,
                forward_schedule.launch_config, 
                backward_schedule.launch_config)
        logger.info("Kernel compiled!")


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

class LoopUnrollConvScatterSum(Convolution):
    def __init__(self, config, idx_dtype=np.int64, torch_op=True):
        assert(torch_op)
        super().__init__(config, idx_dtype, torch_op, deterministic=False)

        self.reference_tp = LoopUnrollTP(config, torch_op=torch_op)
        from fast_tp.implementations.convolution.scatter import scatter_sum
        self.scatter_sum = scatter_sum

    def forward(self, L1_in, L2_in, weights, rows, cols):
        tp_outputs = self.reference_tp(L1_in[cols], L2_in, weights)
        return self.scatter_sum(src=tp_outputs, index=rows, dim=0, dim_size=L1_in.shape[0])

    def forward_cpu(self, L1_in, L2_in, weights, L3_out, graph):
        tp_outputs = np.zeros((graph.nnz, self.L3.dim), dtype=L3_out.dtype)
        self.reference_tp.forward_cpu(L1_in[graph.cols], L2_in, tp_outputs, weights)
        np.add.at(L3_out, graph.rows, tp_outputs)

    def backward_cpu(
            self,
            L1_in : np.ndarray,
            L1_grad : np.ndarray,
            L2_in : np.ndarray,
            L2_grad : np.ndarray,
            L3_grad : np.ndarray,
            weights : np.ndarray,
            weights_grad : np.ndarray,
            graph):
        L1_grad_bcast = np.zeros((graph.nnz, self.L1.dim), dtype=L1_grad.dtype)
        self.reference_tp.backward_cpu(
                L1_in[graph.cols], L1_grad_bcast, L2_in, L2_grad, L3_grad[graph.rows], weights, weights_grad)
        np.add.at(L1_grad, graph.cols, L1_grad_bcast)

    @staticmethod
    def name():
        return "LoopUnrollConvScatterSum" 