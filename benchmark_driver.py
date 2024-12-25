import numpy as np
import numpy.linalg as la
import itertools, typing

from src.benchmark.logging_utils import *

from src.implementations.E3NNTensorProduct import E3NNTensorProduct 
from src.implementations.LoopUnrollTP import LoopUnrollTP
from src.implementations.CUETensorProduct import CUETensorProduct
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from src.benchmark.tpp_creation_utils import *

'''
Paper-ready benchmarks; driver.py is used for prototyping / debugging. 
'''
CTPP = ChannelwiseTPP

mace_conv = [  
    CTPP("128x2e + 128x1o + 128x0e", "1x0e + 1x1e + 1x2e + 1x3e", 2, "mace-large"),
    CTPP("128x1o + 128x0e", "1x0e + 1x1e", 1, "mace-medium"),
]

nequip_conv = [
    CTPP('32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', '0e + 1o + 2e', 2, 'nequip-lips'),
    CTPP('64x0o + 64x0e + 64x1o + 64x1e', '0e + 1o', 1, 'nequip-revmd17-aspirin'),
    CTPP('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e', '0e + 1o + 2e', 2, 'nequip-revmd17-toluene'),
    CTPP('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', 3, 'nequip-revmd17-benzene'),
    CTPP('32x0o + 32x0e + 32x1o + 32x1e', '0e + 1o', 1, 'nequip-waterA'),
    CTPP('32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e + 32x3o + 32x3e', '0e + 1o + 2e + 3o', 3, 'nequip-waterB')
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
    implementations = [ E3NNTensorProduct, 
                        CUETensorProduct, 
                        LoopUnrollTP
                        ]
    directions = ['forward']

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, mace_conv + nequip_conv, directions)]

    # CUE tensor product cannot handle backwards pass for all input configs 
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

if __name__=='__main__':
    benchmark_conv()
    #benchmark_roofline()