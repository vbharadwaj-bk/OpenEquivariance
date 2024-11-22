import numpy as np
from src.implementations.e3nn_lite import *

from src.benchmark.logging_utils import *
logger = getLogger()

class ComputationSchedule:
    def __init__(self, 
            config, 
            smem_limit, 
            warps_per_block, 
            direction,
            irrep_dtype,
            weight_dtype):
        '''
        smem_limit: size of available shared memory in bytes 
        '''
        # Note: does not work with variances for irreps; easy to add that in 

        # Step 1: Break the irreps and the instructions into chunks of at most 32 x 32 x 32. 

        self.L1_raw, self.L2_raw, self.L3_raw = config.irreps_in1, config.irreps_in2, config.irreps_out

        reps_raw = [self.L1_raw, self.L2_raw, self.L3_raw]
        reps = [Irreps(), Irreps(), Irreps()]

        irrep_maps = {} # Maps a (rep_raw #, ir_idx_raw) to a lst[ir_idx]

        for rep_raw_idx, rep in enumerate(reps_raw):
            for ir_idx_raw, mul_ir in enumerate(rep):
                irrep_maps[rep_raw_idx, ir_idx_raw] = []
                for mul_start in range(0, mul_ir.mul, 32): 
                    mul = min(32, mul_ir.mul - mul_start) 
                    reps[rep_raw_idx] += [(mul, mul_ir.ir)]
                    irrep_maps[rep_raw_idx, ir_idx_raw].append(len(reps[rep_raw_idx]) - 1)

        self.new_instructions = []

        for (u, v, w, connection_mode, has_weight, path_weight, path_shape) in config.instructions:
            if connection_mode == "uvu":
                for i, idx1 in enumerate(irrep_maps[0, u]):
                    for idx2 in irrep_maps[1, v]:
                        self.new_instructions.append((idx1, idx2, irrep_maps[2, w][i], connection_mode, has_weight, path_weight))

            else:
                raise Exception(f"Connection mode {connection_mode} not supported!")

        self.new_instructions.sort(key=lambda x: (x[2], x[0], x[1]))

        self.L1, self.L2, self.L3 = reps

        self.updated_config = TPProblem(self.L1, self.L2, self.L3, 
            self.new_instructions, irrep_normalization="none", path_normalization="none", internal_weights=False, shared_weights=config.shared_weights)

        self.new_instructions = self.updated_config.instructions

        assert(self.updated_config.weight_numel == config.weight_numel)

        memory_per_warp = smem_limit // warps_per_block - 8 # 8 bytes for padding

        # Step 2: Loop through the instructions, assigning them to segments that fit into shared memory
        # for a single warp. Could be replaced by a more powerful algorithm. 

        self.segments = []
        cL1, cL2, cL3, cinst = set(), set(), set(), []

        def calculate_forward_smem(L1_set, L2_set, L3_set, inst_idxs):
            smem = 0

            irrep_size = np.dtype(irrep_dtype).itemsize

            smem += sum([self.L1[el].dim for el in L1_set]) * irrep_size 
            smem += sum([self.L2[el].dim for el in L2_set]) * irrep_size
            smem += sum([self.L3[el].dim for el in L3_set]) * irrep_size

            weights_smem = 0
            for inst_idx in inst_idxs:
                inst = self.new_instructions[inst_idx]

                if inst.has_weight:
                    if connection_mode == "uvu":
                        smem += np.prod(inst.path_shape)

            weights_smem *= np.dtype(weight_dtype).itemsize

            return smem

        inst_idx = 0
        while inst_idx <= len(self.new_instructions):
            smem_required = None
            if inst_idx < len(self.new_instructions):
                u, v, w, *others = self.new_instructions[inst_idx]
                smem_required = calculate_forward_smem(cL1 | {u}, cL2 | {v}, cL3 | {w}, cinst + [inst_idx]) 
            else:
                inst_idx += 1

            #print((smem_required, memory_per_warp))

            if inst_idx >= len(self.new_instructions) or smem_required > memory_per_warp:
                if len(cinst) > 0:
                    self.segments.append((cL1, cL2, cL3, cinst))
                    cL1, cL2, cL3, cinst = set(), set(), set(), []
                else:
                    raise Exception("Scheduling failed, memory allocation too small to accomodate segment!")
            else:
                cL1.add(u)
                cL2.add(v)
                cL3.add(w)
                cinst.append(inst_idx)
                inst_idx += 1

        logger.info(f"Scheduling succeeded with {len(self.segments)} segments.")

        print(self.segments)

        for i in range(len(self.segments)):
            L1_idxs, L2_idxs, L3_idxs, inst_idxs = self.segments[i]

            L1Map = IrrepMapping(self.L1, L1_idxs)
            L2Map = IrrepMapping(self.L2, L2_idxs)
            L3Map = IrrepMapping(self.L3, L3_idxs)

            instructions = [
                (L1_map[inst.i_in1], L2_map[inst.i_in2], L3_map[inst.i_out], inst.connection_mode, inst.has_weight, inst.path_weight) 
                    for inst in [self.new_instructions[idx] for idx in inst_idxs]
            ]

            problem = TPProblem(L1sub, L2sub, L3sub, instructions, irrep_normalization="none", path_normalization="none", internal_weights=False, shared_weights=config.shared_weights)


class IrrepMapping:
    '''
    Maps irreps from a source to a destination
    '''
    def __init__(self, src_irreps, idxs):
        self.src_irreps = src_irreps
        self.idxs = sorted(list(idxs))
        self.dst_irreps = Irreps([src_irreps[idx] for idx in self.idxs])
        self.src_dst_map = {idx: i for i, idx in enumerate(self.idxs)}

        src_ranges = [src_irreps.slices()[idx] for idx in self.src_dst_map]
        dst_ranges = [self.dst_irreps.slices()[i] for i in self.src_dst_map.values()]

        print(src_ranges)
        print(dst_ranges)



