import numpy as np
from openequivariance.implementations.TensorProductBase import TensorProductBase

class NumpyTensorProduct(TensorProductBase):
    def __init__(self, config, torch_op=False):
        super().__init__(config, torch_op=torch_op)

        assert(not config.shared_weights)

    def exec_tensor_product(self,
            batch : np.uint64,
            L1_in: np.uint64,
            L2_in: np.uint64,
            L3_out: np.uint64,
            weights: np.uint64):
        raise NotImplementedError("NumpyTensorProduct does not support exec_tensor_product")

    def forward_cpu(self, L1_in, L2_in, L3_out, weights):
        L1, L2, L3 = self.L1, self.L2, self.L3
        config = self.config 
        slices = {  1: L1.slices(), 
                    2: L2.slices(), 
                    3: L3.slices() }

        L3_out[:] = np.zeros((L1_in.shape[0], L3.dim), dtype=np.float32)

        # Should fold this into the tripartite graph class
        weight_counts = [mul for (mul, _) in L3]
        weight_offsets = [0] 
        for count in weight_counts:
            weight_offsets.append(weight_offsets[-1] + count)

        for i in range(len(config.instructions)):
            (irr1, irr2, irr3, _, _, path_weight, _) = config.instructions[i] 
            cg_tensor = self.load_cg_tensor(L1[irr1].ir.l, L2[irr2].ir.l, L3[irr3].ir.l) * path_weight
            
            start1, end1 = slices[1][irr1].start, slices[1][irr1].stop
            start2, end2 = slices[2][irr2].start, slices[2][irr2].stop
            start3, end3 = slices[3][irr3].start, slices[3][irr3].stop

            # Assumes uvu interactions for the weights 
            L3_out[:, start3:end3] += np.einsum('bui,bvj,buv,ijk->buvk', 
                    L1_in[:, start1:end1].reshape((L1_in.shape[0], L1[irr1].mul, L1[irr1].ir.dim)),
                    L2_in[:, start2:end2].reshape((L2_in.shape[0], L2[irr2].mul, L2[irr2].ir.dim)),
                    weights[:, weight_offsets[i]:weight_offsets[i+1]].reshape((L1_in.shape[0], L1[irr1].mul, L2[irr2].mul )),
                    cg_tensor).reshape(L1_in.shape[0], -1)

    def backward_cpu(self, L1_in, L2_in, L3_grad, weights):
        raise NotImplementedError("NumpyTensorProduct does not support backward_cpu")

    @staticmethod
    def name():
        return "NumpyTensorProduct"