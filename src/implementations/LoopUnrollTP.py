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
    def testcases():
        return [("32x5e", "1x5e", "32x3e"),
                ("32x2e", "1x2e", "32x2e"),
                ("32x4e", "1x3e", "32x1e"),
                ("32x4e", "1x3e", "32x5e")]

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out):
        L1, L2, L3 = self.L1, self.L2, self.L3

        def transpose_rep_mult(arr, rep, dir="forward"):
            i_shape = (arr.shape[0], rep.mult(0), 2 * rep.type(0) + 1)

            if dir == "backward":
                i_shape = (i_shape[0], i_shape[2], i_shape[1]) 

            rs1 = arr.reshape(i_shape)
            rs1t = rs1.transpose([0, 2, 1])
            return rs1t.reshape((arr.shape[0], -1)).copy() 

        L1_in_copy = transpose_rep_mult(L1_in, L1, "forward") 
        L2_in_copy = transpose_rep_mult(L2_in, L2, "forward") 
        L3_out_copy = np.zeros_like(L3_out)

        self.internal.exec_tensor_product_cpu(L1_in_copy, L2_in_copy, L3_out_copy) 
        L3_out[:] = transpose_rep_mult(L3_out_copy, L3, "backward")

    @staticmethod
    def name():
        return "LoopUnrollTP"