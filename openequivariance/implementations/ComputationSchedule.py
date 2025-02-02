import numpy as np
from openequivariance.extlib import *
from openequivariance.implementations.e3nn_lite import *
from itertools import accumulate
from openequivariance.benchmark.logging_utils import *
from openequivariance.implementations.TensorProductBase import *
logger = getLogger()

class IrrepMapping:
    '''
    Maps irreps from a source to a destination set.
    '''
    def __init__(self, src_irreps, idxs):
        self.src_irreps = src_irreps
        self.idxs = sorted(list(idxs))
        self.dst_irreps = Irreps([src_irreps[idx] for idx in self.idxs])
        self.src_dst_map = {idx: i for i, idx in enumerate(self.idxs)}

        src_ranges = [src_irreps.slices()[idx] for idx in self.src_dst_map]
        dst_ranges = [self.dst_irreps.slices()[i] for i in self.src_dst_map.values()]

        self.original_src_ranges = src_ranges
        self.original_dst_ranges = dst_ranges

        # Merge adjacent src and dst ranges
        self.src_ranges = []
        self.dst_ranges = []

        src_start, dst_start = src_ranges[0].start, dst_ranges[0].start
        src_end, dst_end = src_ranges[0].stop, dst_ranges[0].stop

        for src_range, dst_range in zip(src_ranges[1:], dst_ranges[1:]):
            if src_range.start == src_end and dst_range.start == dst_end:
                src_end, dst_end = src_range.stop, dst_range.stop
            else:
                self.src_ranges.append(slice(src_start, src_end))
                self.dst_ranges.append(slice(dst_start, dst_end))
                src_start, dst_start = src_range.start, dst_range.start
                src_end, dst_end = src_range.stop, dst_range.stop

        self.src_ranges.append(slice(src_start, src_end))
        self.dst_ranges.append(slice(dst_start, dst_end))
        self.copy_ranges = list(zip(self.src_ranges, self.dst_ranges))

        self.persist_load = False
        self.persist_store = False

        self.storeback_procedure = {idx: "write" for idx in self.idxs}

class CGTensor:
    def __init__(self, l1, l2, l3, normalization_factor, dtype):
        suffix_map = {
            np.float32: "f",
            np.float64: "L"
        }

        tensor = TensorProductBase.load_cg_tensor(l1, l2, l3)
        coord1, coord2, coord3 = [arr.astype(np.int32).copy() for arr in np.nonzero(tensor)]
        float_values = tensor[np.nonzero(tensor)].astype(dtype).copy() * normalization_factor
        values = [str(float.hex(float(val))) + suffix_map[dtype] for val in float_values]

        self.tuples = [(coord1[i], coord2[i], coord3[i], values[i]) for i in range(len(values))]
        self.tuples.sort(key=lambda tup: (tup[1], tup[0], tup[2]))
        self.nnz = len(values)

class ComputationSegment:
    def __init__(self, L1Map, L2Map, L3Map, problem, smem, weight_offset, irrep_dtype):
        self.L1Map = L1Map
        self.L2Map = L2Map
        self.L3Map = L3Map
        self.maps = [L1Map, L2Map, L3Map]

        self.problem = problem
        self.smem = smem
        self.weight_offset = weight_offset # Starting point for weights in overall problem. 

        self.L1 = problem.irreps_in1
        self.L2 = problem.irreps_in2
        self.L3 = problem.irreps_out

        self.interactions = [(u, v, w,
                CGTensor(self.L1[u].ir.l, self.L2[v].ir.l, self.L3[w].ir.l, path_weight, irrep_dtype)) 
                for (u, v, w, _, _, path_weight, _) in problem.instructions]

        #self.interactions.sort(key=lambda x: (x[2], x[0], x[1]))


def create_schedule_case2(instructions, memory_per_warp, calculate_smem):
    segments = []
    cL1 = set([inst[0] for inst in instructions])  
    cL2 = set([inst[1] for inst in instructions])  
    cL3, cinst = set(), []

    inst_idx = 0
    while inst_idx <= len(instructions):
        smem_required = None
        if inst_idx < len(instructions):
            u, v, w, *others = instructions[inst_idx]
            smem_required = calculate_smem(cL1, cL2, cL3 | {w}, cinst + [inst_idx]) 
        else:
            inst_idx += 1

        if inst_idx >= len(instructions) or smem_required["total"] > memory_per_warp:
            if len(cinst) > 0:
                segments.append((cL1, cL2, cL3, cinst))
                cL3, cinst = set(), []
            else:
                raise Exception(f"{direction.title()} scheduling failed, memory allocation too small to accomodate segment!")
        else:
            cL3.add(w)
            cinst.append(inst_idx)
            inst_idx += 1

    return segments

def create_schedule_case3(instructions, memory_per_warp, calculate_smem):
    segments = []
    cL1, cL2, cL3, cinst = set(), set(), set(), []

    inst_idx = 0
    while inst_idx <= len(instructions):
        smem_required = None
        if inst_idx < len(instructions):
            u, v, w, *others = instructions[inst_idx]
            smem_required = calculate_smem(cL1 | {u}, cL2 | {v}, cL3 | {w}, cinst + [inst_idx]) 
        else:
            inst_idx += 1

        if inst_idx >= len(instructions) or smem_required["total"] > memory_per_warp:
            if len(cinst) > 0:
                segments.append((cL1, cL2, cL3, cinst))
                cL1, cL2, cL3, cinst = set(), set(), set(), []
            else:
                raise Exception(f"{direction.title()} scheduling failed, memory allocation too small to accomodate segment!")
        else:
            cL1.add(u)
            cL2.add(v)
            cL3.add(w)
            cinst.append(inst_idx)
            inst_idx += 1

    return segments


class ComputationSchedule:
    def __init__(self, 
            config, 
            smem_limit, 
            warps_per_block,
            block_count, 
            direction,
            irrep_dtype,
            weight_dtype,
            include_scratch=False,
            stream_weights=False, 
            schedule_type=2):
        '''
        smem_limit: size of available shared memory in bytes 
        '''
        # Note: does not work with variances for irreps; easy to add that in 

        # Step 1: Break the irreps and the instructions into chunks of at most warpsize * warpsize * warpsize 

        launch_config = KernelLaunchConfig()
        self.warpsize = launch_config.warp_size
        self.chunksize = self.warpsize
        self.L1_raw, self.L2_raw, self.L3_raw = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.total_warps = warps_per_block * block_count

        dtype_to_str_map = {
            np.float32: "float",
            np.double: "double"
        }

        self.irrep_dtype_cstr = dtype_to_str_map[irrep_dtype]
        self.weight_dtype_cstr = dtype_to_str_map[weight_dtype]

        # Stream weights on the fly before pre-loading 
        self.stream_weights = stream_weights 

        reps_raw = [self.L1_raw, self.L2_raw, self.L3_raw]
        reps = [Irreps(), Irreps(), Irreps()]

        irrep_maps = {} # Maps a (rep_raw #, ir_idx_raw) to a lst[ir_idx]

        for rep_raw_idx, rep in enumerate(reps_raw):
            for ir_idx_raw, mul_ir in enumerate(rep):
                irrep_maps[rep_raw_idx, ir_idx_raw] = []
                for mul_start in range(0, mul_ir.mul, self.warpsize): 
                    mul = min(self.warpsize, mul_ir.mul - mul_start) 
                    reps[rep_raw_idx] += [(mul, mul_ir.ir)]
                    irrep_maps[rep_raw_idx, ir_idx_raw].append(len(reps[rep_raw_idx]) - 1)

        self.new_instructions = []

        for (u, v, w, connection_mode, has_weight, path_weight, path_shape) in config.instructions:
            if connection_mode == "uvu":
                for i, idx1 in enumerate(irrep_maps[0, u]):
                    for idx2 in irrep_maps[1, v]:
                        self.new_instructions.append((idx1, idx2, irrep_maps[2, w][i], connection_mode, has_weight, path_weight ** 2))

            elif connection_mode == "uvw":
                for idx1 in irrep_maps[0, u]:
                    for idx2 in irrep_maps[1, v]:
                        for idx3 in irrep_maps[2, w]:
                            self.new_instructions.append((idx1, idx2, idx3, connection_mode, has_weight, path_weight ** 2))

        #self.new_instructions.sort(key=lambda x: (x[2], x[0], x[1]))

        self.L1, self.L2, self.L3 = reps
        self.updated_config = TPProblem(self.L1, self.L2, self.L3, 
            self.new_instructions, irrep_normalization="none", path_normalization="none", 
            internal_weights=False, shared_weights=config.shared_weights)

        self.new_instructions = self.updated_config.instructions

        assert(self.updated_config.weight_numel == config.weight_numel)

        self.memory_per_warp = smem_limit // warps_per_block
        self.memory_per_warp -= self.memory_per_warp % 4

        # =====================================================================
        # Shared memory partitioning functions 

        def calculate_forward_smem(L1_set, L2_set, L3_set, inst_idxs): 
            irrep_itemsize = np.dtype(irrep_dtype).itemsize
            weight_itemsize = np.dtype(weight_dtype).itemsize
            smem = {
                "L1": {"size": sum([self.L1[el].dim for el in L1_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "L2": {"size": sum([self.L2[el].dim for el in L2_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "L3": {"size": sum([self.L3[el].dim for el in L3_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "weights": {"size": 0, "dtype": self.weight_dtype_cstr},
                "scratch": {"size": 0, "dtype": self.weight_dtype_cstr}
            }

            weights_smem = 0
            for inst_idx in inst_idxs:
                inst = self.new_instructions[inst_idx]

                if inst.has_weight:
                    if inst.connection_mode == "uvu":
                        weights_smem += np.prod(inst.path_shape)

            smem["weights"]["size"] = weights_smem * weight_itemsize 

            if include_scratch: 
                smem["weights"]["size"] = self.warpsize * self.warpsize * weight_itemsize
                # Max irrep size of 10 -> dim = 21 
                smem["scratch"]["size"] = (self.warpsize * 21) * weight_itemsize 

            range_offsets = list(accumulate([smem[name]["size"] for name in smem], initial=0))
            for i, name in enumerate(smem):
                smem[name]["offset"] = range_offsets[i]

            smem["total"] = sum([smem[name]["size"] for name in smem])

            return smem


        def calculate_backward_smem(L1_set, L2_set, L3_set, inst_idxs): 
            irrep_itemsize = np.dtype(irrep_dtype).itemsize
            weight_itemsize = np.dtype(weight_dtype).itemsize
            smem = {
                "L1": {"size": sum([self.L1[el].dim for el in L1_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "L1_grad": {"size": sum([self.L1[el].dim for el in L1_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "L2": {"size": sum([self.L2[el].dim for el in L2_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "L2_grad": {"size": sum([self.L2[el].dim for el in L2_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "L3_grad": {"size": sum([self.L3[el].dim for el in L3_set]) * irrep_itemsize, "dtype": self.irrep_dtype_cstr},
                "weights": {"size": 0, "dtype": self.weight_dtype_cstr},
                "weights_grad": {"size": 0, "dtype": self.weight_dtype_cstr},
                "scratch": {"size": 0, "dtype": self.weight_dtype_cstr}
            }

            weights_smem = 0
            for inst_idx in inst_idxs:
                inst = self.new_instructions[inst_idx]

                if inst.has_weight:
                    if inst.connection_mode == "uvu":
                        weights_smem += np.prod(inst.path_shape)

            smem["weights"]["size"] = weights_smem * np.dtype(weight_dtype).itemsize
            smem["weights_grad"]["size"] = weights_smem * np.dtype(weight_dtype).itemsize

            if include_scratch: 
                smem["weights"]["size"] = self.warpsize * self.warpsize * weight_itemsize
                # We can reuse the weight buffer to accumulate the gradient in shared memory 
                smem["weights_grad"]["size"] = 0 
                # Max irrep size of 10 -> dim = 21 
                smem["scratch"]["size"] = (self.warpsize * 21) * weight_itemsize 

            range_offsets = list(accumulate([smem[name]["size"] for name in smem], initial=0))
            for i, name in enumerate(smem):
                smem[name]["offset"] = range_offsets[i]

            smem["total"] = sum([smem[name]["size"] for name in smem]) 

            return smem

        # =====================================================================

        # Step 2: Loop through the instructions, assigning them to segments that fit into shared memory
        # for a single warp. Could be replaced by a more powerful algorithm. 
        if direction == "forward":
            calculate_smem = calculate_forward_smem
        elif direction == "backward":
            calculate_smem = calculate_backward_smem

        schedule2_succeeded = False
        try:
            if schedule_type != 2:
                raise Exception("Asked for schedule case 3.")
            self.segments = create_schedule_case2(self.new_instructions, self.memory_per_warp, calculate_smem)
            logger.info(f"{direction.title()} case 2 scheduling succeeded with {len(self.segments)} segments.") 
            schedule2_succeeded = True
        except Exception as e:
            self.segments = create_schedule_case3(self.new_instructions, self.memory_per_warp, calculate_smem) 
            logger.info(f"{direction.title()} case 3 scheduling succeeded with {len(self.segments)} segments.")

        for i in range(len(self.segments)):
            L1_idxs, L2_idxs, L3_idxs, inst_idxs = self.segments[i]

            L1Map = IrrepMapping(self.L1, L1_idxs)
            L2Map = IrrepMapping(self.L2, L2_idxs)
            L3Map = IrrepMapping(self.L3, L3_idxs)

            instructions = [
                (L1Map.src_dst_map[inst.i_in1], 
                L2Map.src_dst_map[inst.i_in2], 
                L3Map.src_dst_map[inst.i_out], 
                inst.connection_mode, inst.has_weight, inst.path_weight ** 2) 
                    for inst in [self.new_instructions[idx] for idx in inst_idxs]
            ]

            problem = TPProblem(L1Map.dst_irreps, L2Map.dst_irreps, L3Map.dst_irreps, instructions, 
                    irrep_normalization="none", path_normalization="none", 
                    internal_weights=False, shared_weights=config.shared_weights)

            weight_offset = 0

            if i > 0:
                weight_offset = self.segments[i-1].weight_offset + self.segments[i-1].problem.weight_numel

            self.segments[i] = ComputationSegment(L1Map, L2Map, L3Map, problem, 
                    calculate_smem(L1_idxs, L2_idxs, L3_idxs, inst_idxs), weight_offset, irrep_dtype)

        for ir_idx, ir in enumerate([self.L1, self.L2, self.L3]):
            for i in range(len(ir)):
                irrep_used = False
                for seg in self.segments:    
                    if i in seg.maps[ir_idx].idxs:
                        if irrep_used:
                            seg.maps[ir_idx].storeback_procedure[i] = "accumulate"
                        irrep_used = True

        if schedule2_succeeded:
            # Allow L1 and L2 irreps to persist in shared memory 
            for i, seg in enumerate(self.segments):
                for ir_map in [seg.L1Map, seg.L2Map]:
                    if i > 0:
                        ir_map.persist_load = True
                    if i < len(self.segments) - 1:
                        ir_map.persist_store = True
                    else:
                        for k in ir_map.idxs:
                            ir_map.storeback_procedure[k] = "write"

        true_max_smem = max([seg.smem["total"] for seg in self.segments])
        self.memory_per_warp = true_max_smem

        launch_config.num_blocks = block_count
        launch_config.num_threads = warps_per_block * self.warpsize
        launch_config.smem = self.memory_per_warp * warps_per_block 
        logger.info(f"{direction.title()} pass needs {launch_config.smem // 1000} KB of shared memory.")
        self.launch_config = launch_config