import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class ShuffleReduceTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3, batch_size):
        super().__init__(L1, L2, L3, batch_size)
        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)
        assert(L1.mult(0) == 1 and L2.mult(0) == 1 and L3.mult(0) == 1)

        tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        coord = [arr.astype(np.uint8).copy() for arr in np.nonzero(tensor)]
        values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        nnz = len(self.values)

        warp_length = 32

        assert(L1.get_rep_length() <= warp_length 
                and L2.get_rep_length() <= warp_length
                and L3.get_rep_length() <= warp_length)

        lane_targets = np.zeros(warp_length, dtype=np.uint8)
        # Each lane is an accumulator for one of the outputs of the TP.
        # Currently, we just assign targets cyclically. 
        for i in range(warp_length):
            lane_targets[i] = i % L3.get_rep_length() 

        lane_values = [[] for i in range(warp_length)]        

        # Greedy algorithm, adds to the minimum lane index 
        for i in range(nnz):
            u, v, w = coord[0][i], coord[1][i], coord[2][i]
            min_lane = 0
            for j in range(warp_length):
                if lane_targets[j] == w and len(lane_values[j]) < len(lane_values[min_lane]):
                    min_lane = j

            lane_values[min_lane].append((u, v, values[i]))

        print(lane_values)
        exit(1)

        self.internal = None 

    @staticmethod
    def name():
        return "ShuffleReduceTensorProduct"