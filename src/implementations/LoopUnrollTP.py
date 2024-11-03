import numpy as np
from build.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct, GPUInfo
from src.benchmark.logging_utils import getLogger, bcolors 
from jinja2 import Environment, PackageLoader, FileSystemLoader 

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

class LoopUnrollTP(TensorProduct):
    def __init__(self, config, torch_op=False):
        super().__init__(config, torch_op=torch_op)
        L1, L2, L3 = self.L1, self.L2, self.L3 
        config = self.config

        for (mul, ir) in L1:
            assert(mul == 32)

        for (mul, ir) in L2:
            assert(mul == 1)

        for (mul, ir) in L3:
            assert(mul == 32)

        # =====================================================================
        env = Environment(loader=FileSystemLoader("src/templates"), extensions=['jinja2.ext.do'])
        env.globals['raise'] = raise_helper 
        env.globals['divide'] = divide 
        env.globals['sizeof'] = sizeof 
        template = env.get_template("loop_unroll_multirep.cuh")

        # =====================================================================

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

        # =====================================================================

        self.forward_config = forward_config
        self.backward_config = backward_config 
        load_cg_tensor = self.load_cg_tensor

        class RepData:
            def __init__(self, rep):
                self.num_irreps = len(rep) 
                self.rep_len = rep.dim
                self.irrep_lengths = [r.dim for (_, r) in rep]
                self.mults = [ mul for (mul, _) in rep] 
                
                slices = rep.slices()
                self.offsets = [0] + [s.stop for s in slices]

        class CGTensor:
            def __init__(self, l1, l2, l3, normalization_factor):
                tensor = load_cg_tensor(l1, l2, l3)
                coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy() * normalization_factor
                values = [str(float.hex(float(val))) + "f" for val in float_values]

                self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
                self.tuples.sort(key=lambda tup: (tup[1], tup[0], tup[2]))
                self.nnz = len(values)

        class Weights:
            def __init__(self, config):
                '''
                For now, assumes all "uvu" connections, and that all outputs
                have trainable weights.
                '''
                self.total_len = config.weight_numel 
                self.offsets = [0]
                offset = 0
                for i in range(len(config.instructions)):
                    offset += config.weight_range_and_shape_for_instruction(i)[1]
                    self.offsets.append(offset)

        interactions = [(u, v, w, 
                CGTensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l, path_weight)) 
                for i, (u, v, w, _, _, path_weight, _) in enumerate(config.instructions)]

        interactions.sort(key=lambda x: (x[2], x[0], x[1]))

        self.jit_kernel = template.render(
            L1=L1, L2=L2, L3=RepData(L3),
            config=config,
            weights=Weights(config),
            interactions=interactions,
            forward_config=forward_config,
            backward_config=backward_config
        )

        logger.info("Starting NVRTC")
        self.internal = JITTPImpl(self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out, weights):
        L1, L2, L3 = self.L1, self.L2, self.L3
        logger.warn(f"{bcolors.WARNING}Executing a transpose that is not benchmarked.{bcolors.ENDC}")

        L1Rep, L2Rep, L3Rep = Representation(str(L1)), Representation(str(L2)), Representation(str(L3))

        L1Rep.transpose_irreps_cpu(L1_in, True)
        L2Rep.transpose_irreps_cpu(L2_in, True)

        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights) 

        L1Rep.transpose_irreps_cpu(L1_in, False)
        L2Rep.transpose_irreps_cpu(L2_in, False)
        L3Rep.transpose_irreps_cpu(L3_out, False)

    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        L1_grad = np.zeros_like(L1_in)
        L2_grad = np.zeros_like(L2_in)
        weights_grad = np.zeros_like(weights)

        L1, L2, L3 = self.L1, self.L2, self.L3
        L1Rep, L2Rep, L3Rep = Representation(str(L1)), Representation(str(L2)), Representation(str(L3))

        logger.warn(f"{bcolors.WARNING}Executing a transpose that is not benchmarked.{bcolors.ENDC}")

        L1Rep.transpose_irreps_cpu(L1_in, True)
        L2Rep.transpose_irreps_cpu(L2_in, True)
        L3Rep.transpose_irreps_cpu(L3_grad, True)

        self.internal.backward_cpu(L1_in, L1_grad, 
                L2_in, L2_grad,
                weights, weights_grad, 
                L3_grad)

        L1Rep.transpose_irreps_cpu(L1_grad, False)
        L2Rep.transpose_irreps_cpu(L2_grad, False)

        return L1_grad, L2_grad, weights_grad

    @staticmethod
    def name():
        return "LoopUnrollTP"