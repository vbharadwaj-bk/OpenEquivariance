import numpy as np
import numpy.linalg as la
import itertools, typing

from src.benchmark.logging_utils import *

from src.implementations.E3NNTensorProduct import E3NNTensorProduct 
from src.implementations.LoopUnrollTP import LoopUnrollTP
from src.implementations.CUETensorProduct import CUETensorProduct
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from src.benchmark.tpp_creation_utils import *
from src.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP

'''
Paper-ready benchmarks; driver.py is used for prototyping / debugging. 
'''
CTPP = ChannelwiseTPP

mace_conv = [
    ("128x0e+128x1o+128x2e", "1x0e+1x1o+1x2e+1x3o", "128x0e+128x1o+128x2e+128x3o", 
    "mace-large"),
    ("128x0e+128x1o", "1x0e+1x1o+1x2e+1x3o", "128x0e+128x1o+128x2e", 
    "mace-medium")
]

nequip_conv = [
    ('32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', '0e + 1o + 2e', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', 
            'nequip-lips'),
    ('64x0o + 64x0e + 64x1o + 64x1e', '0e + 1o', '64x0o + 64x0e + 64x1o + 64x1e',
            'nequip-revmd17-aspirin'),
    ('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e', '0e + 1o + 2e', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e',
            'nequip-revmd17-toluene'),
    ('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', 
            'nequip-revmd17-benzene'),
    ('32x0o + 32x0e + 32x1o + 32x1e', '0e + 1o', '32x0o + 32x0e + 32x1o + 32x1e', 
            'nequip-waterA'),
    #CTPP('32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e + 32x3o + 32x3e', '0e + 1o + 2e + 3o', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e + 32x3o + 32x3e',
    #        'nequip-waterB')
]


roofline_configs = [
    SingleInstruction(L1, L2, L3, cm, f"[{i+1}]#{L1} x {L2} -> {L3} ({cm})")
    for i, (L1, L2, L3, cm) in enumerate([
        ("32x1e", "1x1e", "32x1e", "uvu"), 
        ("32x2e", "1x1e", "32x2e", "uvu"),
        ("32x3e", "1x3e", "32x3e", "uvu"),
        ("32x5e", "1x5e", "32x3e", "uvu"),
        ("32x5e", "1x3e", "32x5e", "uvu") 
    ])
]

def benchmark_conv():
    implementations = [ E3NNTensorProductCompiled,
                        CUETensorProduct,
                        LoopUnrollTP
                        ]
    directions = ['forward', 'backward']

    problems = []
    for config in mace_conv + nequip_conv:
        problem32 = CTPP(*config)
        problems.append(problem32)

        problem64 = CTPP(*config)
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64
        problems.append(problem64)
 
    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    # CUE tensor product cannot handle backwards pass for all input configs 
    tests = [test for test in tests 
            if test.direction == 'forward' 
            or test.implementation != CUETensorProduct
            or 'mace' in test.problem.label]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_iter=5,
        bench_batch_size=50000,
        prng_seed=11111
    )

    logger.setLevel(logging.INFO)
    bench_suite.run(tests)

def benchmark_roofline():
    implementations = [LoopUnrollTP, CUETensorProduct]
    directions = ['forward', 'backward']

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, roofline_configs, directions)]

    # CUE tensor product cannot handle backwards pass 
    tests = [test for test in tests 
            if test.direction == 'forward' 
            or test.implementation != CUETensorProduct]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_iter=5,
        bench_batch_size=50000,
        prng_seed=11111
    )

    logger.setLevel(logging.INFO)
    bench_suite.run(tests)

def benchmark_fully_connected():
    FCTPP = FullyConnectedTPProblem
    implementations = [
        LoopUnrollTP,
        #MultiplicityOuterProductTP,
        E3NNTensorProduct]

    directions = ['forward', 'backward']

    problems = [
            FCTPP("2x1e", "2x1e", "2x1e"),
            FCTPP("2x4e", "2x4e", "2x4e"),
            FCTPP("2x8e", "2x8e", "2x8e"),

            FCTPP("4x1e", "4x1e", "4x1e"),
            FCTPP("4x4e", "4x4e", "4x4e"),
            FCTPP("4x8e", "4x8e", "4x8e"),

            FCTPP("8x1e", "8x1e", "8x1e"),
            FCTPP("8x4e", "8x4e", "8x4e"),
            FCTPP("8x8e", "8x8e", "8x8e"),

            FCTPP("16x1e", "16x1e", "16x1e"),
            FCTPP("16x4e", "16x4e", "16x4e"),
            FCTPP("16x8e", "16x8e", "16x8e"),

            FCTPP("32x1e", "32x1e", "32x1e"),
            FCTPP("32x4e", "32x4e", "32x4e"), 
    ]

    for problem in problems:
        problem.label = f"({str(problem.irreps_in1)}) x {str(problem.irreps_in2)} -> {str(problem.irreps_out)}"

    tests = [TestDefinition(implementation, problem, direction, 
                correctness=True, benchmark=True) 
             for problem, direction, implementation
             in itertools.product(problems, directions, implementations)]
 
    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        correctness_batch_size=1000,
        bench_batch_size=100000,
        prng_seed=11111,
        torch_op=True)

    logger.setLevel(logging.INFO)
    bench_suite.run(tests)

if __name__=='__main__':
    benchmark_conv()
    #benchmark_roofline()
    #benchmark_fully_connected()