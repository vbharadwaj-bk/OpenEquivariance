
import itertools
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from src.benchmark.logging_utils import *

from src.implementations.E3NNTensorProduct import E3NNTensorProduct 
from src.implementations.LoopUnrollTP import LoopUnrollTP
from src.implementations.CUETensorProduct import CUETensorProduct
from src.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from src.benchmark.tpp_creation_utils import FullyConnectedTPProblem

logger = getLogger()

if __name__ == '__main__':
    FCTPP = FullyConnectedTPProblem

    problems = [
            FCTPP("2x1e", "2x1e", "2x1e"),
            FCTPP("4x1e", "4x1e", "4x1e"),
            FCTPP("8x1e", "8x1e", "8x1e"),
            FCTPP("16x1e", "16x1e", "16x1e"),
            FCTPP("32x1e", "32x1e", "32x1e"),
        ]

    directions : list[Direction] = [
        'forward',
        'backward',
    ]

    implementations = [
            E3NNTensorProduct,
            CUETensorProduct, 
            MultiplicityOuterProductTP
        ]

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
                for problem, direction, implementation
                in itertools.product(problems, directions, implementations)]

    bench_suite = TestBenchmarkSuite(
            correctness_threshold = 5e-5,
            num_iter=5,
            bench_batch_size=50000,
            prng_seed=11111
        )
    
    logger.setLevel(logging.INFO)
    bench_suite.run(tests)