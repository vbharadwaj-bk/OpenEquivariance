import e3nn
from e3nn import o3

import sys, os
from build.kernel_wrapper import *
from src.implementations.LoopUnrollTP import *

def compare_output_to_e3nn(config, batch_size):
    assert(isinstance(config[2], int))

    reps = RepTriple(Representation(config[0]), Representation(config[1]), 3)
    instructions = [reps.interactions(i) for i in range(reps.num_interactions())]
    instructions = [(i[0], i[1], i[2], "uvu", False) for i in instructions]

    irreps_L1 = o3.Irreps(reps.L1.to_string())
    irreps_L2 = o3.Irreps(reps.L2.to_string())
    irreps_L3 = o3.Irreps(reps.L3.to_string()) 

    fast_tp = LoopUnrollTP(reps, batch_size)
    tp = o3.TensorProduct(irreps_L1, irreps_L2, irreps_L3, instructions=instructions) # internal_weights=False, shared_weights=False
    print(tp.weight_numel)

    In1 = irreps_L1.randn(batch_size, -1)
    In2 = irreps_L2.randn(batch_size, -1)
    L3_out_fast = np.zeros((1, triple.L3.get_rep_length() ), dtype=np.float32)

    fast_tp.exec_tensor_product_cpu(In1.numpy(), In2.numpy(), L3_out_fast)
    e3nn_output = tp(In1, In2)

    print(e3nn_output / (L3_out_fast * np.sqrt(5))) 