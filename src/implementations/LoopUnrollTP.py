import numpy as np
from build.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct, GPUInfo
from src.benchmark.logging_utils import getLogger, bcolors 
from jinja2 import Environment, PackageLoader, FileSystemLoader 

logger = getLogger()

#def sizeof(dtype):
#    if dtype in ["double"]:
#        return 8
#    elif dtype in ["float", "int", "unsigned int"]:
#        return 4
#    elif dtype in ["char"]:
#        return 1

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
        #env.filters['sizeof'] = sizeof 
        template = env.get_template("loop_unroll_multirep.cuh")

        config = KernelLaunchConfig()
        config.num_blocks = GPUInfo.A100_SMS * 4 
        # Warning: correctness check fail at 1024 threads 
        config.num_threads = 512
        config.smem = 163840

        self.launch_config = config

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
                self.coord1, self.coord2, self.coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
                float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
                self.values = [str(float.hex(float(val))) + "f" for val in float_values]
                self.nnz = len(self.values)

        interactions = [reps.interactions(i) for i in range(reps.num_interactions())]
        interactions = [(u, v, w, CGTensor(L1.type(u), L2.type(v), L3.type(w))) for u, v, w in interactions] 

        self.jit_kernel = template.render(
            L1=RepData(L1), L2=RepData(L2), L3=RepData(L3),
            interactions=interactions,
            thread_block_size = config.num_threads
        )
        #print(self.jit_kernel)

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