from src.implementations.E3NNTensorProduct import *

from src.benchmark.logging_utils import *
from build.kernel_wrapper import *
from src.benchmark.TestBenchmarkSuite import *
from src.implementations.LoopUnrollTP import *
from src.implementations.ManyOneUVWTP import *
from src.implementations.NumpyTensorProduct import *

from src.implementations.e3nn_lite import *

import numpy as np
import numpy.linalg as la

logger = getLogger()

def debug(tp_impl, config, direction="forward"): 
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out 
    batch_size = 1

    tp = tp_impl(config)

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.dim)), dtype=np.float32)
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.dim)), dtype=np.float32)
    weights = np.array(rng.uniform(size=(batch_size, config.weight_numel)), dtype=np.float32) 

    weights[:] = 1.0

    L3_out = np.zeros((batch_size, L3.dim), dtype=np.float32)

    if direction == "forward":
        tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)
        _, ground_truth = tp.test_correctness(L1_in, L2_in, weights, L3_out, reference_implementation=E3NNTensorProduct)
        print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))
        #print(L3_out / ground_truth)
        print(L3_out)
        print(ground_truth)
        #print(L1_in)

    elif direction == "backward":
        L3_grad = L3_out
        L3_grad[:] = rng.uniform(size=(batch_size, L3.dim)) 
        L1_grad, L2_grad, weights_grad = tp.backward_cpu(L1_in, L2_in, L3_grad, weights)


        reference = E3NNTensorProduct(config)
        L1_grad_ref, L2_grad_ref, weights_grad_ref = reference.backward_cpu(L1_in, L2_in, L3_grad, weights)

        print(la.norm((L1_grad-L1_grad_ref).flatten(), ord=np.inf))
        print(la.norm((L2_grad-L2_grad_ref).flatten(), ord=np.inf))
        print(la.norm((weights_grad-weights_grad_ref).flatten(), ord=np.inf))

    else:
        assert(False)

if __name__=='__main__':
    configs = [
        single_inst_conf("32x1e", "32x5e", "32x5e", "uvw", True),
        #single_inst_conf("32x5e", "1x5e", "32x3e", "uvu", True),
        #mace_conf("32x3e + 32x2e", "1x0e + 1x1e", 3),
        #mace_conf("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3),
        #mace_conf("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]

    throughput_configs = [
        mace_conf(f"{i}x2e + {i}x1e + {i}x0e", "1x0e + 1x1e", 3)
        for i in range(1, 32, 2)
    ]

    bench_suite = TestBenchmarkSuite(configs, bench_batch_size=10000)
    bench_suite.run([ManyOneUVWTP], direction="forward", reference_impl=None)

    #debug(ManyOneUVWTP, configs[0], direction="forward")