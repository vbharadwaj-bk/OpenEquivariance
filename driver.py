import json, os, time, pathlib
import cppimport
import cppimport.import_hook
cppimport.settings["use_filelock"] = False
from scipy.sparse import coo_matrix
import numpy as np
import numpy.linalg as la

from src.wrapper.kernel_wrapper import *
from src.implementations.GemmTP import *

class TestBenchmarkSuite:
    def __init__(self):
        self.configs = \
            [(5, 5, 3),
             (2, 2, 2),
             (4, 3, 1),
             (4, 3, 5)
            ]

        self.num_warmup = 10
        self.num_iter = 30
        self.correctness_batch_size = 100000
        self.bench_batch_size = 10000000
        self.prng_seed = 12345

    def run(self, tp_implementations, correctness=True):
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)
        metadata = {
            "configs": self.configs,
            "implementations": [impl.name() for impl in tp_implementations]
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f) 

        for config in self.configs: 
            L1, L2, L3 = config

            for impl in tp_implementations:
                tp = impl(L1, L2, L3)

                # No need to regenerate, but this will do for now 
                rng = np.random.default_rng(self.prng_seed)
                L1_in  = np.array(rng.uniform(size=(self.correctness_batch_size, tp.get_row_length(1))), dtype=np.float32) 
                L2_in  = np.array(rng.uniform(size=(self.correctness_batch_size, tp.get_row_length(2))), dtype=np.float32)
                L3_out = np.zeros((self.correctness_batch_size, tp.get_row_length(3)), dtype=np.float32)

                print("Started correctness check!")
                tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
                correctness, ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)
                print("Finished Correctness Check!")

                print("Started benchmark!")
                benchmark = tp.benchmark(self.num_warmup, self.num_iter, self.bench_batch_size, prng_seed=self.prng_seed) 
                print("Completed benchmark!")

                result = {
                    "config": config,
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark
                }
                fname = pathlib.Path(f"{output_folder}/{L1}_{L2}_{L3}_{impl.name()}.json")

                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

def debug(tp_impl, config):
    L1, L2, L3 = config
    tp = tp_impl(L1, L2, L3)
    batch_size = 1

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, tp.get_row_length(1))), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(batch_size, tp.get_row_length(2))), dtype=np.float32)
    L3_out = np.zeros((batch_size, tp.get_row_length(3)), dtype=np.float32)

    tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
    correctness, ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)

    print(L3_out)
    print(ground_truth)

if __name__=='__main__':
    #bench_suite = TestBenchmarkSuite()
    #bench_suite.run([ThreadTensorProduct])
    debug(ThreadTensorProduct, (4, 3, 5))
