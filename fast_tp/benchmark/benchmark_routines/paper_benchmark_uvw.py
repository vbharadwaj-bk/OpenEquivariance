import itertools
import sys
import os
import logging
import copy
import pathlib
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from fast_tp.benchmark.logging_utils import getLogger
from fast_tp.implementations.E3NNTensorProduct import E3NNTensorProductCompiledCUDAGraphs
from fast_tp.implementations.CUETensorProduct import CUETensorProduct
from fast_tp.implementations.LoopUnrollTP import LoopUnrollTP
from fast_tp.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from fast_tp.benchmark.tpp_creation_utils import FullyConnectedTPProblem
from fast_tp.benchmark.benchmark_configs import e3nn_torch_tetris_polynomial, diffdock_configs

logger = getLogger()

def run_paper_uvw_benchmark() -> pathlib.Path:

    FCTPP = FullyConnectedTPProblem

    problems =  list(itertools.chain(
        e3nn_torch_tetris_polynomial,
        diffdock_configs,
    ))

    directions : list[Direction] = [
        'forward',
        'backward',
    ]

    float64_problems = copy.deepcopy(problems)
    for problem in float64_problems: 
        problem.irrep_dtype = np.float64
        problem.weight_dtype = np.float64
    
    problems += float64_problems

    implementations = [
        E3NNTensorProductCompiledCUDAGraphs,
        CUETensorProduct,  
        LoopUnrollTP,
        ]

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
                for problem, direction, implementation
                in itertools.product(problems, directions, implementations)]

    bench_suite = TestBenchmarkSuite(
            correctness_threshold = 5e-5,
            num_warmup=100,
            num_iter=100,
            bench_batch_size=50_000,
            prng_seed=11111
        )
    
    logger.setLevel(logging.INFO)
    return bench_suite.run(tests)

if __name__ == '__main__':
    run_paper_uvw_benchmark()