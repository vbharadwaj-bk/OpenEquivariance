import numpy as np
from build.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

from jinja2 import Environment, PackageLoader, FileSystemLoader 

class LoopUnrollTP(TensorProduct):
    def __init__(self, L1, L2, L3, batch_size):
        super().__init__(L1, L2, L3, batch_size)
        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        assert(L1.mult(0) == 32 and L2.mult(0) == 1 and L3.mult(0) == 32)

        tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        coord = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
        float_values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        str_values = [str(float.hex(float(val))) + "f" for val in float_values] 

        # =====================================================================
        env = Environment(loader=FileSystemLoader("src/templates"))
        template = env.get_template("loop_unroll.cuh")

        self.jit_kernel = template.render(
            L1_one_rep_len=L1.type(0) * 2 + 1,
            L2_one_rep_len=L2.type(0) * 2 + 1,
            L3_one_rep_len=L3.type(0) * 2 + 1,

            L1_stride = L1.get_rep_length(),
            L2_stride = L2.get_rep_length(),
            L3_stride = L3.get_rep_length(),

            L1_mult = L1.mult(0),
            L2_mult = L2.mult(0),
            L3_mult = L3.mult(0),

            nnz = len(str_values),
            values = str_values,
            coord1 = coord[0],
            coord2 = coord[1],
            coord3 = coord[2]
        ) 

        self.internal = UnrollTPImpl(L1, L2, L3, self.jit_kernel) 

    @staticmethod
    def name():
        return "LoopUnrollTP"