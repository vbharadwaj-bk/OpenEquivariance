
import itertools
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from src.benchmark.logging_utils import *

from src.implementations.E3NNTensorProduct import E3NNTensorProduct 
from src.implementations.ManyOneUVWTP import ManyOneUVWTP
from src.implementations.CUETensorProduct import CUETensorProduct
from src.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP
from src.implementations.LoopReorderUVWTP import LoopReorderUVWTP
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from src.benchmark.tpp_creation_utils import FullyConnectedTPProblem
from src.benchmark.benchmark_configs import e3nn_torch_tetris, e3nn_torch_tetris_polynomial

logger = getLogger()

if __name__ == '__main__':
    FCTPP = FullyConnectedTPProblem

    problems =  list(itertools.chain(
        e3nn_torch_tetris_polynomial,
    ))

    directions : list[Direction] = [
        'forward',
        'backward',
    ]

    implementations = [
            E3NNTensorProduct,
            # CUETensorProduct, 
            MultiplicityOuterProductTP,
            # LoopReorderUVWTP,
        ]

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
                for problem, direction, implementation
                in itertools.product(problems, directions, implementations)]

    bench_suite = TestBenchmarkSuite(
            correctness_threshold = 5e-5,
            num_iter=5,
            bench_batch_size=500_000,
            prng_seed=11111
        )
    
    logger.setLevel(logging.INFO)
    bench_suite.run(tests)