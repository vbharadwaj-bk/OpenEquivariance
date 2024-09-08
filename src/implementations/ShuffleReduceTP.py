import numpy as np
import cppimport
import cppimport.import_hook
from src.wrapper.kernel_wrapper import *
from src.implementations.TensorProduct import TensorProduct

class ShuffleReduceTensorProduct(TensorProduct):
    def __init__(self, L1, L2, L3, batch_size, mode="opt"):
        super().__init__(L1, L2, L3, batch_size)

        assert(L1.num_irreps() == 1 and L2.num_irreps() == 1 and L3.num_irreps() == 1)

        # To begin, we only handle one combo and multiplicity 1. 
        assert(L1.mult(0) == 1 and L2.mult(0) == 1 and L3.mult(0) == 1)
        assert(mode == "opt" or mode == "prototype")

        tensor = self.load_cg_tensor(L1.type(0), L2.type(0), L3.type(0))
        coord = [arr.astype(np.uint8).copy() for arr in np.nonzero(tensor)]
        values = tensor[np.nonzero(tensor)].astype(np.float32).copy()
        nnz = len(values)

        warp_length = 32

        assert(L1.get_rep_length() <= warp_length 
                and L2.get_rep_length() <= warp_length
                and L3.get_rep_length() <= warp_length)

        lane_targets = np.zeros(warp_length, dtype=np.uint8) # The output index that each lane must accumulate to 
        lanes_by_target = [[] for _ in range(L3.get_rep_length())] # The list of lanes used as accumulators for each output index 

        # Each lane is an accumulator for one of the outputs of the TP.
        # Currently, we just assign targets cyclically. 
        for i in range(warp_length):
            target = i % L3.get_rep_length()
            lane_targets[i] = target 
            lanes_by_target[target].append(i) 

        for target in lanes_by_target:
            target.sort()

        lanes = [[] for _ in range(warp_length)]        

        # Greedy algorithm, adds to the minimum lane index 
        for i in range(nnz):
            u, v, w = coord[0][i], coord[1][i], coord[2][i]
            min_lane = 0
            for j in range(warp_length):
                if lane_targets[j] == w and len(lanes[j]) < len(lanes[min_lane]):
                    min_lane = j

            lanes[min_lane].append((u, v, values[i]))

        max_lane_length = np.max([len(lane) for lane in lanes])

        reduction_depth = int(np.max([len(target) for target in lanes_by_target])).bit_length()
        warp_values = np.zeros((max_lane_length, warp_length), dtype=np.float32)

        # Can probably smash these values into a uin64_t mask 
        l1_indices  = np.zeros((max_lane_length, warp_length), dtype=np.uint8) 
        l2_indices  = np.zeros((max_lane_length, warp_length), dtype=np.uint8)
        red_lanes = np.zeros((reduction_depth, warp_length), dtype=np.uint8)

        for i, lane in enumerate(lanes):
            for j, (u, v, value) in enumerate(lane): 
                warp_values[j][i] = value
                l1_indices[j][i] = u
                l2_indices[j][i] = v


        for d in range(reduction_depth):
            jump = 2 ** d
            for target in lanes_by_target:
                for j in target:
                    red_lanes[d][j] = target[0] 

                for i in range(0, len(target), jump * 2):
                    j = target[i]

                    if i + jump < len(target):
                        red_lanes[d][j] = target[i + jump]


        self.warp_length, self.reduction_depth, self.max_lane_length = \
                warp_length, reduction_depth, max_lane_length
        self.warp_values, self.l1_indices, self.l2_indices, self.red_lanes = \
                warp_values, l1_indices, l2_indices, red_lanes

        self.internal = None 
        self.mode = mode


    def prototype(self, L1_in, L2_in, L3_out):
        '''
        Tests variable setup and the algorithm logic. 
        '''
        L1, L2, L3 = self.L1, self.L2, self.L3
        warp_values, l1_indices, l2_indices, red_lanes = \
                self.warp_values, self.l1_indices, self.l2_indices, self.red_lanes
        batch_size = L1_in.shape[0]

        def shuffle(vec, src):
            return vec[src]

        for i in range(batch_size):

            # Step 1: load vectors into warp lanes 
            vec1, vec2, vec3 = [np.zeros(self.warp_length)]
            vec1[:L1.get_rep_length()] = L1_in[i, :]
            vec2[:L2.get_rep_length()] = L2_in[i, :]

            for j in range(self.max_lane_length):
                for tid in range(self.warp_length):
                    wval = self.warp_values[j][tid]
                    l1_val = shuffle(vec1, self.l1_indices[j][tid])
                    l2_val = shuffle(vec2, self.l2_indices[j][tid])
                    vec3[tid] += l1_val * l2_val * wval 

            vec3_copy = vec3.copy()  
            for tid in range(self.warp_length):
                if tid < L3.get_rep_length():
                    vec3[tid] = 0 

            for j in range(self.reduction_depth):
                for tid in range(self.warp_length):
                    vec3[tid] += shuffle(vec3, self.red_lanes[j][tid])

            vec3 += vec3_copy
            L3_out[i, :] = vec3[:L3.get_rep_length()]


    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out):
        if mode == "proto":
            self.prototype(L1_in, L2_in, L3_out)
        elif mode == "opt": 
            self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out)

    @staticmethod
    def name():
        return "ShuffleReduceTensorProduct"