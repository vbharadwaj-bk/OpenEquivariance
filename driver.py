import json, os, time, pathlib 

from src.benchmark.logging_utils import *
from build.kernel_wrapper import *
from src.implementations.GemmTP import *
from src.implementations.ThreadTP import *
from src.implementations.ShuffleReduceTP import *
from src.implementations.LoopUnrollTP import *

import numpy as np
import numpy.linalg as la

logger = getLogger()

def config_to_rep_triple(config):
    reps = None 
    if isinstance(config[0], tuple):
        reps = [Representation(config[i][0], config[i][1]) for i in range(3)]
    elif isinstance(config[0], str) and not isinstance(config[2], int):
        reps = [Representation(config[i]) for i in range(3)]
    elif isinstance(config[0], str) and isinstance(config[2], int):
        return RepTriple(Representation(config[0]), Representation(config[1]), config[2])
    return RepTriple(reps[0], reps[1], reps[2])

class TestBenchmarkSuite:
    def __init__(self, configs,
        num_warmup = 10,
        num_iter = 30,
        correctness_batch_size = 10000,
        bench_batch_size = 10000000,
        prng_seed = 12345
    ):
        self.configs = configs 
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.correctness_batch_size = correctness_batch_size 
        self.bench_batch_size = bench_batch_size 
        self.prng_seed = 12345

    def run(self, tp_implementations, correctness=True):        
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        rep_sets = [config_to_rep_triple(config) for config in self.configs] 
        metadata = {
            "configs": [set.to_string() for set in rep_sets], 
            "implementations": [impl.name() for impl in tp_implementations]
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        for reps in rep_sets:
            L1, L2, L3 = reps.L1, reps.L2, reps.L3 
            rng = np.random.default_rng(self.prng_seed)
            L1_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L1.get_rep_length())), dtype=np.float32) 
            L2_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L2.get_rep_length())), dtype=np.float32) 
            L3_out = np.zeros((self.correctness_batch_size, L3.get_rep_length()), dtype=np.float32)
            for impl in tp_implementations:
                tc_name = f"{reps.to_string()}, {impl.name()}"
                logger.info(f'Starting {tc_name}.')

                tp_correctness = impl(reps, self.correctness_batch_size)
                tp_bench = impl(reps, self.bench_batch_size)

                tp_correctness.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
                correctness, _ = tp_correctness.test_correctness(L1_in, L2_in, L3_out)

                benchmark = tp_bench.benchmark(self.num_warmup, self.num_iter, self.bench_batch_size, prng_seed=self.prng_seed) 
                rnames= [rep.to_string().replace(' ', '') for rep in [L1, L2, L3]]
                result = {
                    "config": rnames, 
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark
                }
         
                fname = pathlib.Path(f"{output_folder}/{rnames[0]}_{rnames[1]}_{rnames[2]}_{impl.name()}.json")

                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f'Finished {tc_name}.')

def debug(tp_impl, config):
    reps = config_to_rep_triple(config)
    L1, L2, L3 = reps.L1, reps.L2, reps.L3
    batch_size = 10000 
    tp = tp_impl(reps, batch_size) 

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.get_rep_length())), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.get_rep_length())), dtype=np.float32) 
    L3_out = np.zeros((batch_size, L3.get_rep_length() ), dtype=np.float32)

    tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
    _ , ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)

    #print(L3_out) 
    #print(ground_truth) 
    #print(L3_out - ground_truth)
    print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))

if __name__=='__main__':
    default_tests = [
            ((1, 5), (1, 5), (1, 3)),
            ((1, 2), (1, 2), (1, 2)),
            ((1, 4), (1, 3), (1, 1)),
            ((1, 4), (1, 3), (1, 5))]

    multiplicity_tests = [
            ((2, 4), (2, 3), (4, 5)),
            ((2, 4), (1, 3), (2, 5)),
            ((1, 4), (2, 3), (2, 5))
    ]

    full_decomp_tests = [
        ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3)#, # Last value is Lmax
        #("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 4),
        #("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]

    bench_suite = TestBenchmarkSuite(full_decomp_tests, bench_batch_size=1000000)
    bench_suite.run([LoopUnrollTP])

    #bench_suite = TestBenchmarkSuite(default_tests, bench_batch_size=32000000)
    #bench_suite.run([ThreadTensorProduct, GemmTensorProduct, ShuffleReduceTensorProduct])

    #bench_suite = TestBenchmarkSuite(default_tests)
    #bench_suite.run([ThreadTensorProduct,
    #                    GemmTensorProduct,
    #                    ShuffleReduceTensorProduct])

    #debug(LoopUnrollTP, ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3))
    #debug(LoopUnrollTP, ("32x4e", "1x3e", "32x5e"))
