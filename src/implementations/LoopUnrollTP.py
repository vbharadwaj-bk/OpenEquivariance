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
        config = self.config

        for (mul, ir) in L2:
            assert(mul == 1)

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_batch.cuh")
        env.globals['enumerate'] = enumerate 

        dp = DeviceProp(0)

        forward_schedule = ComputationSchedule(config, 
                smem_limit=80000, warps_per_block=6,
                block_count=dp.multiprocessorCount * 2,
                direction = "forward",
                irrep_dtype = np.float32,
                weight_dtype = np.float32
        )

        backward_config = KernelLaunchConfig()
        backward_config.num_blocks = dp.multiprocessorCount * 4
        backward_config.num_threads = 128
        backward_config.smem = (2 * L1.dim + 2 * L2.dim + 2 * config.weight_numel + L3.dim)  * sizeof("float") * backward_config.num_threads // backward_config.warp_size
        logger.info(f"Backward pass needs {backward_config.smem // 1000} KB of shared memory.")

        #if backward_config.smem > dp.maxSharedMemPerBlock:
        #    raise Exception(f"Error, requested shared memory {backward_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")

        # =====================================================================

        self.backward_config = backward_config 

        class CGTensor:
            def __init__(self, l1, l2, l3, normalization_factor):
                tensor = TensorProduct.load_cg_tensor(l1, l2, l3)
                coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy() * normalization_factor
                values = [str(float.hex(float(val))) + "f" for val in float_values]

                self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
                self.tuples.sort(key=lambda tup: (tup[1], tup[0], tup[2]))
                self.nnz = len(values)

        interactions = [(u, v, w, i, 
                CGTensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l, path_weight)) 
                for i, (u, v, w, _, _, path_weight, _) in enumerate(config.instructions)]

        interactions.sort(key=lambda x: (x[2], x[0], x[1]))

        self.jit_kernel = template.render(
            L1=L1, L2=L2, L3=L3,
            config=config,
            interactions=interactions,
            forward_config=forward_schedule.launch_config,
            backward_config=backward_config,
            forward_schedule=forward_schedule
        )

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, forward_schedule.launch_config, self.backward_config)
        logger.info("Kernel compiled!")

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
