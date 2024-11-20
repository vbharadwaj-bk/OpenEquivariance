import numpy as np
from src.implementations.e3nn_lite import *

class ComputationScheduler:
    def __init__(self, config):
        self.L1_raw, self.L2_raw, self.L3_raw = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.L1, self.L2, self.L3 = Irreps(), Irreps(), Irreps()

        reps_raw = [self.L1_raw, self.L2_raw, self.L3_raw]
        reps = [self.L1, self.L2, self.L3]

        irrep_maps = {} # Maps a (rep_raw #, ir_idx_raw) to a (rep #, lst[ir_idx])

        for rep_raw_idx, rep in enumerate(reps_raw):
            for ir_idx_raw, mul_ir in enumerate(rep):
                irrep_maps[rep_raw_idx, ir_idx_raw] = []
                for mul_start in range(0, mul_ir.mul, 32): 
                    mul = min(32, mul_ir.mul - mul_start) 
                    reps[rep_raw_idx] += [(mul, mul_ir.ir)]
                    irrep_maps[rep_raw_idx, ir_idx_raw].append(len(reps[rep_raw_idx]) - 1)

        print(irrep_maps)