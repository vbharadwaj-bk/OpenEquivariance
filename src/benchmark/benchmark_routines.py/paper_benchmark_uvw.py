
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
from src.benchmark.benchmark_configs import e3nn_torch_tetris, e3nn_docs_tetris

logger = getLogger()

if __name__ == '__main__':
    FCTPP = FullyConnectedTPProblem

    # problems = [
    #         # FCTPP("2x1e", "2x1e", "2x1e"),
    #         # FCTPP("2x4e", "2x4e", "2x4e"),
    #         # FCTPP("2x8e", "2x8e", "2x8e"),

    #         # FCTPP("4x1e", "4x1e", "4x1e"),
    #         # FCTPP("4x4e", "4x4e", "4x4e"),
    #         # FCTPP("4x8e", "4x8e", "4x8e"),

    #         # FCTPP("8x1e", "8x1e", "8x1e"),
    #         # FCTPP("8x4e", "8x4e", "8x4e"),
    #         # FCTPP("8x8e", "8x8e", "8x8e"),

    #         # FCTPP("16x1e", "16x1e", "16x1e"),
    #         # FCTPP("16x4e", "16x4e", "16x4e"),
    #         # FCTPP("16x8e", "16x8e", "16x8e"),

    #         FCTPP("32x1e", "32x1e", "32x1e"),
    #         FCTPP("32x4e", "32x4e", "32x4e"),
    #         # FCTPP("32x8e", "32x8e", "32x8e"),

    #         FCTPP("8x1e + 8x1e + 8x1e + 8x1e", "31x1e", "32x1e"),
    #         FCTPP("8x4e + 8x4e + 8x4e + 8x4e", "31x4e", "32x4e"),
    #         # FCTPP("8x8e + 8x8e + 8x8e + 8x8e", "31x8e", "32x8e"),
    #     ]

    problems =  list(itertools.chain(
        e3nn_docs_tetris,
        # e3nn_torch_tetris,
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