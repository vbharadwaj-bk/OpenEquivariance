import numpy as np
from build.kernel_wrapper import *
from src.templates.jinja_utils import *
from src.implementations.TensorProduct import TensorProduct 
from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

class ManyOneUVWTP(TensorProduct):
    def __init__(self, config, torch_op=False):
        super().__init__(config, torch_op=torch_op)
        L1, L2, L3 = self.L1, self.L2, self.L3 
        config = self.config

        mul_0 = L1[0].mul
        assert(mul_0 == 32)

        for (mul, ir) in L1:
            assert(mul == mul_0) 

        for (mul, ir) in L3:
            assert(mul == mul_0)

        assert(len(config.instructions) == 1)
        #assert(config.instructions[0].connection_mode == "uvw")

        env = get_jinja_environment()
        template = env.get_template("many_one_uvw.cuh")

        dp = DeviceProp(0)

        forward_config = KernelLaunchConfig()
        forward_config.num_blocks = dp.multiprocessorCount * 4
        forward_config.num_threads = 256
        forward_config.smem = (L1.dim + L2.dim + L3.dim + 32 * 32 + 32 * 12)  * sizeof("float") * forward_config.num_threads // forward_config.warp_size  
        logger.info(f"Forward pass needs {forward_config.smem // 1000} KB of shared memory.")

        if forward_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {forward_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")

        backward_config = KernelLaunchConfig()
        backward_config.num_blocks = dp.multiprocessorCount * 4
        backward_config.num_threads = 192
        backward_config.smem = (2 * L1.dim + 2 * L2.dim + 2 * 32 * 32 + L3.dim)  * sizeof("float") * backward_config.num_threads // backward_config.warp_size
        logger.info(f"Backward pass needs {backward_config.smem // 1000} KB of shared memory.")

        if backward_config.smem > dp.maxSharedMemPerBlock:
            raise Exception(f"Error, requested shared memory {backward_config.smem}B hits or exceeds maximum, {dp.maxSharedMemPerBlock}B !")

        # =====================================================================

        self.forward_config = forward_config
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
            forward_config=forward_config,
            backward_config=backward_config
        )

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out, weights):
        L1, L2, L3 = self.L1, self.L2, self.L3
        logger.warning(f"{bcolors.WARNING}Executing a transpose that is not benchmarked.{bcolors.ENDC}")

        L1Rep, L2Rep, L3Rep = Representation(str(L1)), Representation(str(L2)), Representation(str(L3))

        L1Rep.transpose_irreps_cpu(L1_in, True)
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights) 
        L1Rep.transpose_irreps_cpu(L1_in, False)

    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        raise NotImplementedError("Backward pass not implemented for ManyOneUVWTP")

    @staticmethod
    def name():
        return "ManyOneUVWTP"