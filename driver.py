import json, os, time, pathlib
import cppimport
import cppimport.import_hook
cppimport.settings["use_filelock"] = False

from src.wrapper.kernel_wrapper import *
from src.implementations.GemmTP import *
from src.implementations.ThreadTP import *
from src.implementations.ShuffleReduceTP import *

import numpy as np
import numpy.linalg as la

def config_to_reps(config):
    return [Representation(config[i][0], config[i][1]) for i in range(3)]

class TestBenchmarkSuite:
    def __init__(self):
        self.configs = \
            [((1, 5), (1, 5), (1, 3)),
             ((1, 2), (1, 2), (1, 2)),
             ((1, 4), (1, 3), (1, 1)),
             ((1, 4), (1, 3), (1, 5))
            ] # Multiplicity, irrep-type pairs

        self.num_warmup = 10
        self.num_iter = 30
        self.correctness_batch_size = 100000
        self.bench_batch_size = 10000000
        self.prng_seed = 12345

    def run(self, tp_implementations, correctness=True):        
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        rep_sets = [config_to_reps(config) for config in self.configs] 
        metadata = {
            "configs": [[rep.to_string() for rep in reps] for reps in rep_sets], 
            "implementations": [impl.name() for impl in tp_implementations]
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        for (L1, L2, L3) in rep_sets: 
            rng = np.random.default_rng(self.prng_seed)
            L1_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L1.get_rep_length())), dtype=np.float32) 
            L2_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L2.get_rep_length())), dtype=np.float32) 
            L3_out = np.zeros((self.correctness_batch_size, L3.get_rep_length()), dtype=np.float32)

            for impl in tp_implementations:
                tp_correctness = impl(L1, L2, L3, self.correctness_batch_size)
                tp_bench = impl(L1, L2, L3, self.bench_batch_size)

                print("Started correctness check!")
                tp_correctness.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
                correctness, ground_truth = tp_correctness.test_correctness(L1_in, L2_in, L3_out)
                print("Finished Correctness Check!")

                print("Started benchmark!")
                benchmark = tp_bench.benchmark(self.num_warmup, self.num_iter, self.bench_batch_size, prng_seed=self.prng_seed) 
                print("Completed benchmark!")

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


def debug(tp_impl, config):
    L1, L2, L3 = config_to_reps(config)
    batch_size = 10
    tp = tp_impl(L1, L2, L3, batch_size)

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.get_rep_length())), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.get_rep_length())), dtype=np.float32) 
    L3_out = np.zeros((batch_size, L3.get_rep_length()), dtype=np.float32)

    tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
    correctness, ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)

    print(L3_out)
    print(ground_truth)

if __name__=='__main__':
    #bench_suite = TestBenchmarkSuite()
    #bench_suite.run([ThreadTensorProduct, GemmTensorProduct])
    debug(ShuffleReduceTensorProduct, ((1, 3), (1, 3), (1, 4)))
