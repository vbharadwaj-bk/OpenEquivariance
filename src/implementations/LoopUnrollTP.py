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
    def __init__(self, reps, batch_size):
        super().__init__(reps, batch_size)
        L1, L2, L3 = self.L1, self.L2, self.L3

        for i in range(L1.num_irreps()):
            assert(L1.mult(i) == 32)

        for i in range(L2.num_irreps()):
            assert(L2.mult(i) == 1)

        for i in range(L3.num_irreps()):
            assert(L3.mult(i) == 32)

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
        forward_config.smem = (L1.get_rep_length() + L2.get_rep_length() + L3.get_rep_length())  * sizeof("float") * forward_config.num_threads // forward_config.warp_size 

        if forward_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {forward_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")


        backward_config = KernelLaunchConfig()
        backward_config.num_blocks = GPUInfo.A100_SMS * 4
        backward_config.num_threads = 256
        backward_config.smem = (2 * L1.get_rep_length() + 2 * L2.get_rep_length() + 2 * reps.num_trainable_weights() + L3.get_rep_length())  * sizeof("float") * backward_config.num_threads // backward_config.warp_size 

        if backward_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {forward_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

        # =====================================================================

        self.forward_config = forward_config
        self.backward_config = backward_config 
        load_cg_tensor = self.load_cg_tensor

        class RepData:
            def __init__(self, rep):
                self.num_irreps = rep.num_irreps()
                self.rep_len = rep.get_rep_length()
                self.irrep_lengths = [rep.type(i) * 2 + 1 for i in range(self.num_irreps)]
                self.mults = [ rep.mult(i) for i in range(self.num_irreps)]
                self.offsets = rep.get_irrep_offsets()

        class CGTensor:
            def __init__(self, l1, l2, l3):
                tensor = load_cg_tensor(l1, l2, l3)
                coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
                values = [str(float.hex(float(val))) + "f" for val in float_values]

                self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
                self.tuples.sort(key=lambda tup: (tup[1], tup[0], tup[2]))
                self.nnz = len(values)

        class Weights:
            def __init__(self, reps):
                '''
                For now, assumes all "uvu" connections, and that all outputs
                have trainable weights.
                '''
                weight_counts = [reps.L3.mult(i) for i in range(reps.L3.num_irreps())]
                self.total_len = sum(weight_counts)                 
                self.offsets = [0]
                offset = 0
                for count in weight_counts:
                    offset += count
                    self.offsets.append(offset) 

        interactions = [reps.interactions(i) for i in range(reps.num_interactions())]
        interactions = [(u, v, w, CGTensor(L1.type(u), L2.type(v), L3.type(w))) for u, v, w in interactions]
        interactions.sort(key=lambda x: (x[2], x[0], x[1]))

        self.jit_kernel = template.render(
            L1=RepData(L1), L2=RepData(L2), L3=RepData(L3),
            weights=Weights(reps),
            interactions=interactions,
            forward_config=forward_config,
            backward_config=backward_config
        )

        logger.info("Starting NVRTC")
        self.internal = UnrollTPImpl(self.reps, self.jit_kernel, self.forward_config, self.backward_config)
        logger.info("Kernel compiled!")

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out):
        L1, L2, L3 = self.L1, self.L2, self.L3
        logger.warn(f"{bcolors.WARNING}Executing a transpose that is not benchmarked.{bcolors.ENDC}")

        L1.transpose_irreps_cpu(L1_in, True)
        L2.transpose_irreps_cpu(L2_in, True)

        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out) 

        L1.transpose_irreps_cpu(L1_in, False)
        L2.transpose_irreps_cpu(L2_in, False)
        L3.transpose_irreps_cpu(L3_out, False)

    @staticmethod
    def name():
        return "LoopUnrollTP"