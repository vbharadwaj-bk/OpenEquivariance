import numpy as np
from build.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct, GPUInfo
from src.benchmark.logging_utils import getLogger, bcolors 
from jinja2 import Environment, PackageLoader, FileSystemLoader 

logger = getLogger()

class LoopUnrollTP(TensorProduct):
    def __init__(self, reps, batch_size):
        super().__init__(reps, batch_size)
        L1, L2, L3 = self.L1, self.L2, self.L3
        #assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        assert(L1.mult(0) == 32 and L2.mult(0) == 1 and L3.mult(0) == 32)

        tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        coord = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
        float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        str_values = [str(float.hex(float(val))) + "f" for val in float_values] 

        # =====================================================================
        env = Environment(loader=FileSystemLoader("src/templates"))
        template = env.get_template("loop_unroll_multirep.cuh")

        config = KernelLaunchConfig()
        config.num_blocks = GPUInfo.A100_SMS * 4 
        # Warning: correctness check fail at 1024 threads 
        config.num_threads = 512
        self.launch_config = config

        class RepData:
            def __init__(self, rep):
                self.one_rep_len = rep.type(0) * 2 + 1 
                self.rep_len = rep.get_rep_length()
                self.mult = rep.mult(0)

        self.jit_kernel = template.render(
            L1=Repdata(L1), L2=RepData(L2), L3=RepData(L3),

            nnz = len(str_values),
            values = str_values,
            coord1 = coord[0],
            coord2 = coord[1],
            coord3 = coord[2],

            thread_block_size = config.num_threads
        ) 

        self.internal = UnrollTPImpl(self.reps, self.jit_kernel, self.launch_config)

    @staticmethod
    def testcases():
        return [("32x5e", "1x5e", "32x3e"),
                ("32x2e", "1x2e", "32x2e"),
                ("32x4e", "1x3e", "32x1e"),
                ("32x4e", "1x3e", "32x5e")]

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