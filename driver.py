import itertools

import numpy as np
import numpy.linalg as la

import torch
import e3nn
from e3nn.o3 import Irrep, Irreps

from src.benchmark.logging_utils import *
from src.benchmark.e3nn_tp_utils import *
from build.kernel_wrapper import *
from src.implementations.GemmTP import *
from src.implementations.ThreadTP import *
from src.implementations.ShuffleReduceTP import *
from src.implementations.LoopUnrollTP import *
from src.implementations.MultiplicityOuterProductTP import *
from src.benchmark.benchmark_suite import TestBenchmarkSuite

logger = getLogger()

def debug(tp_impl, config : e3nn.o3.TensorProduct, direction="forward"):
    # THESE ARE NOW E3NN IRREPS
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out 
    batch_size = 1
    tp = tp_impl(config)

    rng = np.random.default_rng(12345)

    if direction == "forward":
        L1_in, L2_in, weights, L3_out = get_random_forward_supplies(config, batch_size, prng_seed=12345)

        tp.exec_tensor_product_cpu(L1_in, L2_in, weights, L3_out)

        ground_truth_out = tp.e3nn_tp(
            torch.Tensor(L1_in),
            torch.Tensor(L2_in),
            torch.Tensor(weights)
            ).numpy(force=True)

        print(L3_out - ground_truth_out)

    elif direction == "backward":
        L1_in, L2_in, L3_grad, weights, weights_grad, L1_in_grad, L2_in_grad = get_random_backward_supplies(config, batch_size, prng_seed=12345)


        tp.backward_cpu(L1_in, L1_in_grad, L2_in, L2_in_grad, L3_grad, weights, weights_grad)

        torch_L1_in = torch.Tensor(L1_in, requires_grad=True)
        torch_L2_in = torch.Tensor(L2_in, requires_grad=True)
        torch_weights = torch.Tennsor(weights, requires_grad=True)

        torch_out = tp.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        torch_out.backward(L3_grad)

        ground_truth_L1_in_grad = torch_L1_in.grad.numpy(force=True)
        ground_truth_L2_in_grad = torch_L2_in.grad.numpy(force=True)
        ground_truth_out_weights_grad = torch_weights.grad.numpy(force=True)

        print(L1_in_grad - ground_truth_L1_in_grad)
        print(L2_in_grad - ground_truth_L2_in_grad)
        print(weights_grad - ground_truth_out_weights_grad)
    else:
        assert(False)

if __name__=='__main__':
    FCTP = e3nn.o3.FullyConnectedTensorProduct
    default_tests = [ FCTP(in1, in2, out) for in1, in2, out in        
        [
        ("1x5e", "1x5e", "1x3e"),
        ("1x2e", "1x2e", "1x2e"),
        ("1x4e", "1x3e", "1x1e"),
        ("1x4e", "1x3e", "1x5e"),
        ]
    ]

    multiplicity_tests = [ FCTP(in1, in2, out) for in1, in2, out in 
        [
        ("1x4e", "1x3e", "1x3e"),
        ("2x4e", "1x3e", "2x5e"),
        ("1x4e", "2x3e", "2x5e"),
        ("2x4e", "2x3e", "4x5e"),
        ]
    ]

    limited_decomp_tests = [ FCTP(in1, in2, out) for in1, in2, out in
        [
        ("32x5e", "1x5e", "32x3e"),
        ("32x3e + 32x2e", "1x0e + 1x1e", (32 * Irreps.spherical_harmonics(3, p=1)).sort().irreps.simplify()),
        ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", (32 * Irreps.spherical_harmonics(3, p=1)).sort().irreps.simplify()), 
        ("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", (32 * Irreps.spherical_harmonics(3, p=1)).sort().irreps.simplify())
        ]
    ]

    bench_suite = TestBenchmarkSuite()

    tests = limited_decomp_tests
    implementations = [MultiplicityOuterProductTP]
    directions = ["forward", "backward"]
    do_correctness = [True] 
    do_benchmark = [True]

    test_list = list(itertools.product(tests, implementations, directions, do_correctness, do_benchmark))

    bench_suite.run(test_list=test_list)

    #debug(LoopUnrollTP, ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3))
    #debug(LoopUnrollTP, ("32x5e", "1x5e", "32x3e"), direction="backward")
