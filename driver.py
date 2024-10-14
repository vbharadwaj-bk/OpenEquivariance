import json, os, time, pathlib 

from src.benchmark.logging_utils import *
from build.kernel_wrapper import *
from src.implementations.GemmTP import *
from src.implementations.ThreadTP import *
from src.implementations.ShuffleReduceTP import *

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
    def __init__(self):
        self.configs = [
            ((1, 5), (1, 5), (1, 3)),
            ((1, 2), (1, 2), (1, 2)),
            ((1, 4), (1, 3), (1, 1)),
            ((1, 4), (1, 3), (1, 5)),

            #((2, 4), (2, 3), (4, 5)),
            #((2, 4), (1, 3), (2, 5)),
            #((1, 4), (2, 3), (2, 5)),
            ] # Multiplicity, irrep-type pairs

        self.num_warmup = 10
        self.num_iter = 30
        self.correctness_batch_size = 100000
        self.bench_batch_size = 10000000
        self.prng_seed = 12345

    def run(self, tp_implementations, direction, correctness=True):        
        assert(direction == "forward" or direction == "backward")

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
                tc_name = f"{reps.to_string()}, {impl.name()}, {direction}"
                logger.info(f'Starting {tc_name}.')

                if correctness and direction == "forward":
                    tp_correctness = impl(reps, self.correctness_batch_size)
                    tp_correctness.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
                    correctness, _ = tp_correctness.test_correctness(L1_in, L2_in, L3_out)

                tp_bench = impl(reps, self.bench_batch_size)
                benchmark = tp_bench.benchmark(self.num_warmup, self.num_iter, self.bench_batch_size, direction, prng_seed=self.prng_seed) 
                rnames= [rep.to_string().replace(' ', '') for rep in [L1, L2, L3]]
                result = {
                    "config": rnames,
                    "direction": direction, 
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark
                }
         
                fname = pathlib.Path(f"{output_folder}/{rnames[0]}_{rnames[1]}_{rnames[2]}_{impl.name()}.json")

                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f'Finished {tc_name}.')

def debug(tp_impl, config, direction="forward"):
    reps = config_to_rep_triple(config)
    L1, L2, L3 = reps.L1, reps.L2, reps.L3
    batch_size = 1
    tp = tp_impl(reps, batch_size)

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.get_rep_length())), dtype=np.float32)
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.get_rep_length())), dtype=np.float32)
    L3_out = np.zeros((batch_size, L3.get_rep_length() ), dtype=np.float32)

    if direction == "forward":
        tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
        _ , ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)
        print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))
    elif direction == "backward":
        L3_grad = L3_out
        L3_grad[:] = rng.uniform(size=(batch_size, L3.get_rep_length())) 
        weights = np.array(rng.uniform(size=(batch_size, reps.num_trainable_weights())), dtype=np.float32)
        L1_grad, L2_grad, weights_grad = tp.backward_cpu(L1_in, L2_in, L3_grad, weights)
        print(L1_grad)
        print(L2_grad)
        print(weights_grad)
    else:
        assert(False)

if __name__=='__main__':
    bench_suite = TestBenchmarkSuite()
    bench_suite.run([
        ThreadTensorProduct, 
        GemmTensorProduct,
        ShuffleReduceTensorProduct
        ])
    #debug(ShuffleReduceTensorProduct, ((1, 4), (1, 3), (1, 5)))
