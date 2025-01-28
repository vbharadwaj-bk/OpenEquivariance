import itertools
import logging 

import numpy as np
import numpy.linalg as la

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.kernel_wrapper import DeviceProp
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct, E3NNTensorProductCompiledCUDAGraphs, E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.benchmark.tpp_creation_utils import ChannelwiseTPP, FullyConnectedTPProblem, SingleInstruction
from openequivariance.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP
from openequivariance.benchmark.benchmark_routines.paper_benchmark_uvw import run_paper_uvw_benchmark

'''
Paper-ready benchmarks; driver.py is used for prototyping / debugging. 
'''

logger = getLogger()

CTPP = ChannelwiseTPP
FCTPP = FullyConnectedTPProblem

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
            'nequip-water'),
    #CTPP('32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e + 32x3o + 32x3e', '0e + 1o + 2e + 3o', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e + 32x3o + 32x3e',
    #        'nequip-waterB')
]


roofline_configs = [
    SingleInstruction(L1, L2, L3, cm, f"[{i+1}]#{L1} x {L2} -> {L3} ({cm})")
    for i, (L1, L2, L3, cm) in enumerate([
        ("128x1e", "1x1e", "128x1e", "uvu"), 
        ("128x2e", "1x1e", "128x2e", "uvu"),
        ("128x3e", "1x3e", "128x3e", "uvu"),
        ("128x5e", "1x5e", "128x3e", "uvu"),
        ("128x5e", "1x3e", "128x5e", "uvu"),
        ("128x6e", "1x3e", "128x6e", "uvu"),
        ("128x7e", "1x4e", "128x7e", "uvu"),
        ("128x7e", "1x7e", "128x7e", "uvu"),
    ])
]

def benchmark_conv():
    implementations = [ 
        E3NNTensorProductCompiledMaxAutotuneCUDAGraphs, 
        CUETensorProduct, 
        LoopUnrollTP,
        ]
    
    directions = [
        'forward', 
        'backward',
        ]

    problems = []
    for config in mace_conv + nequip_conv:
        problem32 = CTPP(*config)
        problems.append(problem32)

        problem64 = CTPP(*config)
        problem64.irrep_dtype = np.float64
        problem64.weight_dtype = np.float64
        problems.append(problem64)
 
    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    # Handle the float64 Benzene case specially
    # since we run out of memory with torch compile
    tests = [test for test in tests
            if 'benzene' not in test.problem.label
            or test.implementation != E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
            or test.problem.irrep_dtype != np.float64]

    tests.extend([TestDefinition(E3NNTensorProduct, 
        CTPP('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', 
                'nequip-revmd17-benzene', irrep_dtype=np.float64, weight_dtype=np.float64), direction, correctness=False, benchmark=True) 
                for direction in ['forward', 'backward']])
    
    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=50000,
        prng_seed=11111
    )

    logger.setLevel(logging.INFO)
    bench_suite.run(tests)

def benchmark_roofline():
    implementations =   [LoopUnrollTP, 
                        CUETensorProduct
                        ]
    directions = [  'forward',
                    'backward']

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, roofline_configs, directions)]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=200000,
        prng_seed=11111,
        torch_op=False
    )

    logger.setLevel(logging.INFO)
    bench_suite.run(tests)

if __name__=='__main__':
    dp = DeviceProp(0)

    paper_benchmark_gpu = "NVIDIA A100-SXM4-80GB"
    if dp.name != paper_benchmark_gpu:
        logger.warning(msg=f"Notice: current GPU ({dp.name}) is not the {paper_benchmark_gpu} used in the paper. Your benchmarks may differ from our reported results.")

    #benchmark_conv()
    #benchmark_roofline()
    run_paper_uvw_benchmark()