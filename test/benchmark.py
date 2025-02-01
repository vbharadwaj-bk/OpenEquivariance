import itertools, logging, argparse

import numpy as np
import numpy.linalg as la

from openequivariance.benchmark.logging_utils import getLogger
from openequivariance.extlib import DeviceProp
from openequivariance.implementations.E3NNTensorProduct import E3NNTensorProduct, E3NNTensorProductCompiledCUDAGraphs, E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP
from openequivariance.implementations.CUETensorProduct import CUETensorProduct
from openequivariance.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from openequivariance.benchmark.tpp_creation_utils import ChannelwiseTPP, FullyConnectedTPProblem, SingleInstruction
from openequivariance.benchmark.benchmark_routines.paper_benchmark_uvw import run_paper_uvw_benchmark

logger = getLogger()

CTPP = ChannelwiseTPP
FCTPP = FullyConnectedTPProblem

implementation_map = {
    'e3nn': E3NNTensorProduct, 
    'e3nn_compiled': E3NNTensorProductCompiledMaxAutotuneCUDAGraphs,
    'cue': CUETensorProduct,
    'oeq': LoopUnrollTP
}

datatype_map = {
    'float32': np.float32,
    'float64': np.float64
}

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

def benchmark_uvu(params):
    implementations = [
        implementation_map[impl] for impl in params.implementations
    ] 
    directions = params.directions
    datatypes = [datatype_map[dt] for dt in params.datatypes]

    problems = []
    for dtype in datatypes:
        for config in mace_conv + nequip_conv:
            problem = CTPP(*config) # float32 by default

            if dtype == np.float64:
                problem.irrep_dtype = np.float64
                problem.weight_dtype = np.float64

            problems.append(problem)
 
    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    # Handle the float64 Benzene case specially
    # since we run out of memory with torch compile
    tests = [test for test in tests
            if 'benzene' not in test.problem.label
            or test.implementation != E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
            or test.problem.irrep_dtype != np.float64]

    if 'e3nn' in params.implementations and 'float64' in params.datatypes:
        tests.extend([TestDefinition(E3NNTensorProduct, 
            CTPP('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', 
                    'nequip-revmd17-benzene', irrep_dtype=np.float64, weight_dtype=np.float64), direction, correctness=False, benchmark=True) 
                    for direction in ['forward', 'backward']])

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=params.batch_size,
        prng_seed=11111
    )

    logger.setLevel(logging.INFO)
    bench_suite.run(tests, params.output_folder)

def benchmark_roofline(params):
    implementations =   [LoopUnrollTP, 
                        CUETensorProduct]
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
    bench_suite.run(tests, params.output_folder)

def correctness(params):
    implementations = [LoopUnrollTP]
    directions = [ 'forward', 
                    #'backward' Disabled temporarily while testing HIP warp reduction
                ]
    problems = [CTPP(*config) for config in mace_conv + nequip_conv]

    tests = [TestDefinition(implementation, problem, direction, correctness=True, benchmark=False) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        prng_seed=11111,
        torch_op=False
    )

    logger.setLevel(logging.INFO)
    bench_suite.run(tests, params.output_folder)

if __name__=='__main__':
    dp = DeviceProp(0)
    paper_benchmark_gpu = "NVIDIA A100-SXM4-80GB"
    if dp.name != paper_benchmark_gpu:
        logger.warning(msg=f"Current GPU ({dp.name}) is not the {paper_benchmark_gpu} used in the paper. Runtime benchmarks may differ from our reported results.")
    parser = argparse.ArgumentParser(description='Benchmark openequivariance kernels')
    parser.add_argument("--output_folder", "-o", type=str, default=None, help="Output folder for benchmark results")

    subparsers = parser.add_subparsers(help='subcommand help', required=True)
    parser_uvu = subparsers.add_parser('uvu', help='Run the UVU kernel benchmark without fusion') 
    parser_uvu.add_argument("--batch_size", "-b", type=int, default=50000, help="Batch size for benchmark")
    parser_uvu.add_argument("--implementations", "-i", type=str, nargs='+', 
            default=['e3nn_compiled', 'cue', 'oeq'], help="Implementations to benchmark",
            choices=['e3nn', 'e3nn_compiled', 'cue', 'oeq'])
    parser_uvu.add_argument("--directions", "-d", type=str, nargs='+',
            default=['forward', 'backward'], help="Directions to benchmark",
            choices=['forward', 'backward'])
    parser_uvu.add_argument("--datatypes", "-t", type=str, nargs='+',
            default=['float32', 'float64'], help="Data types to benchmark",
            choices=['float32', 'float64'])
    parser_uvu.set_defaults(func=benchmark_uvu)

    parser_roofline = subparsers.add_parser('roofline', help='Run the roofline comparison')
    parser_roofline.set_defaults(func=benchmark_roofline)

    parser_correctness = subparsers.add_parser('correctness', help='Run correctness tests')
    parser_correctness.set_defaults(func=correctness)

    args = parser.parse_args()
    args.func(args)

    #run_paper_uvw_benchmark()
