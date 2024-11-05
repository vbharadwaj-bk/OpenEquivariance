from src.implementations.Convolution import *
from src.implementations.TensorProduct import GPUInfo
from src.templates.jinja_utils import *
from build.kernel_wrapper import *

class LoopUnrollConv(Convolution):
    def __init__(self, config):
        super().__init__(config)
        L1, L2, L3 = self.L1, self.L2, self.L3 
        config = self.config

        env = get_jinja_environment()
        template = env.get_template("loop_unroll_conv.cuh") 

        forward_config = KernelLaunchConfig()
        forward_config.num_blocks = GPUInfo.A100_SMS * 4
        forward_config.num_threads = 256
        forward_config.smem = (L1.dim + L2.dim + L3.dim + config.weight_numel)  * sizeof("float") * forward_config.num_threads // forward_config.warp_size 
        logger.info(f"Forward pass needs {forward_config.smem // 1000} KB of shared memory.")

        if forward_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {forward_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

        backward_config = KernelLaunchConfig()
        backward_config.num_blocks = GPUInfo.A100_SMS * 4
        backward_config.num_threads = 192
        backward_config.smem = (2 * L1.dim + 2 * L2.dim + 2 * config.weight_numel + L3.dim)  * sizeof("float") * backward_config.num_threads // backward_config.warp_size
        logger.info(f"Backward pass needs {backward_config.smem // 1000} KB of shared memory.")

        if backward_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {backward_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

        self.forward_config = forward_config
        self.backward_config = backward_config

        self.jit_kernel = template.render(
            L1=L1, L2=L2, L3=L3,
            config=config,
            # interactions=interactions,
            forward_config=forward_config,
            backward_config=backward_config
        ) 

        self.internal = JITConvImpl(self.jit_kernel, forward_config, backward_config)

    @staticmethod
    def name():
        return "LoopUnrollConv"