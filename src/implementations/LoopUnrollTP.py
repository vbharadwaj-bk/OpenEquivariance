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
    def __init__(self, reps, torch_op=False):
        super().__init__(reps, torch_op=torch_op)
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
        forward_config.smem = (L1.get_rep_length() + L2.get_rep_length() + L3.get_rep_length() + reps.num_trainable_weights())  * sizeof("float") * forward_config.num_threads // forward_config.warp_size 
        logger.info(f"Forward pass needs {forward_config.smem} bytes of shared memory.")

        if forward_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {forward_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

        backward_config = KernelLaunchConfig()
        backward_config.num_blocks = GPUInfo.A100_SMS * 4
        backward_config.num_threads = 192
        backward_config.smem = (2 * L1.get_rep_length() + 2 * L2.get_rep_length() + 2 * reps.num_trainable_weights() + L3.get_rep_length())  * sizeof("float") * backward_config.num_threads // backward_config.warp_size
        logger.info(f"Backward pass needs {backward_config.smem} bytes of shared memory.")

        if backward_config.smem > GPUInfo.max_smem:
            raise Exception(f"Error, requested shared memory {backward_config.smem}B hits or exceeds maximum, {GPUInfo.max_smem}B !")

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
            def __init__(self, l1, l2, l3, normalization_factor):
                tensor = load_cg_tensor(l1, l2, l3)
                coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy() * normalization_factor
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
                assert(reps.num_trainable_weights() == self.total_len) 
                self.offsets = [0]
                offset = 0
                for count in weight_counts:
                    offset += count
                    self.offsets.append(offset)

        # --------- Hasty attempt at normalization coefficients --------------
        irrep_normalization, path_normalization = 'component', 'element'
        interactions = [reps.interactions(i) for i in range(reps.num_interactions())]
        normalization_coefficients = []

        def dim(mul_ir):
            return mul_ir[1] * 2 + 1

        #def num_elements(connection_mode):
        #    return {
        #        "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
        #        "uvu": self.irreps_in2[ins.i_in2].mul,
        #        "uvv": self.irreps_in1[ins.i_in1].mul,
        #        "uuw": self.irreps_in1[ins.i_in1].mul,
        #        "uuu": 1,
        #        "uvuv": 1,
        #        "uvu<v": 1,
        #        "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        #    }[connection_mode]

        for (u, v, w) in interactions:
            mul_ir_in1 = (L1.mult(u), L1.type(u))
            mul_ir_in2 = (L2.mult(v), L2.type(v))
            mul_ir_out = (L3.mult(w), L3.type(w)) 
            #assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            #assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            #assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw']

            if irrep_normalization == 'component':
                alpha = dim(mul_ir_out)
            if irrep_normalization == 'norm':
                alpha = dim(mul_ir_in1) * dim(mul_ir_in2) 
            if irrep_normalization == 'none':
                alpha = 1

            if path_normalization == 'element':
                x = sum(
                    1.0 #in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements("uvu")
                    for (u_other, v_other, w_other) in interactions 
                    if w_other == w 
                )
            #elif path_normalization == 'path':
            #    x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements("uvu")
            #    x *= len([i for i in instructions if i.i_out == ins.i_out])
            #elif path_normalization == 'none':
            #    x = 1

            if x > 0.0:
                alpha /= x

            alpha *= 1.0 # out_var[ins.i_out]

            # For now, we don't allow manually setting path weights
            # alpha *= ins.path_weight

            normalization_coefficients += [np.sqrt(alpha)]

        print(normalization_coefficients)

        interactions = [(u, v, w, 
                CGTensor(L1.type(u), L2.type(v), L3.type(w), normalization_coefficients[i])) 
                for i, (u, v, w) in enumerate(interactions)]
        # --------------------------------------------------------------------


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

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out, weights):
        L1, L2, L3 = self.L1, self.L2, self.L3
        logger.warn(f"{bcolors.WARNING}Executing a transpose that is not benchmarked.{bcolors.ENDC}")

        L1.transpose_irreps_cpu(L1_in, True)
        L2.transpose_irreps_cpu(L2_in, True)

        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights) 

        L1.transpose_irreps_cpu(L1_in, False)
        L2.transpose_irreps_cpu(L2_in, False)
        L3.transpose_irreps_cpu(L3_out, False)

    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        L1_grad = np.zeros_like(L1_in)
        L2_grad = np.zeros_like(L2_in)
        weights_grad = np.zeros_like(weights)

        L1, L2, L3 = self.L1, self.L2, self.L3
        logger.warn(f"{bcolors.WARNING}Executing a transpose that is not benchmarked.{bcolors.ENDC}")

        L1.transpose_irreps_cpu(L1_in, True)
        L2.transpose_irreps_cpu(L2_in, True)
        L3.transpose_irreps_cpu(L3_grad, True)

        self.internal.backward_cpu(L1_in, L1_grad, 
                L2_in, L2_grad,
                weights, weights_grad, 
                L3_grad)

        L1.transpose_irreps_cpu(L1_grad, False)
        L2.transpose_irreps_cpu(L2_grad, False)

        return L1_grad, L2_grad, weights_grad

    @staticmethod
    def name():
        return "LoopUnrollTP"