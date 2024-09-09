import json, os, time, pathlib
import cppimport
import cppimport.import_hook
cppimport.settings["use_filelock"] = False

from src.wrapper.kernel_wrapper import *
from src.implementations.GemmTP import *
from src.implementations.ThreadTP import *

import numpy as np
import numpy.linalg as la
import cupy as cp

def config_to_reps(config):
    return [Representation(config[i][0], config[i][1]) for i in range(3)]

class TestBenchmarkSuite:
    def __init__(self):
        self.configs = [
            ((1, 5), (1, 5), (1, 3)),
            ((1, 2), (1, 2), (1, 2)),
            ((1, 4), (1, 3), (1, 1)),
            ((1, 4), (1, 3), (1, 5)),
            ((2, 4), (2, 3), (4, 5)),
            ((2, 4), (1, 3), (2, 5)),
            ((1, 4), (2, 3), (2, 5)),
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
            L1_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L1.mult(0), 2 * L1.type(0) + 1)), dtype=np.float32) 
            L2_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L2.mult(0), 2 * L2.type(0) + 1)), dtype=np.float32) 
            L3_out = np.zeros((self.correctness_batch_size, L3.mult(0), 2 * L3.type(0) + 1), dtype=np.float32)
            for impl in tp_implementations:
                print(f"({L1.to_string()})x({L2.to_string()})->({L3.to_string()}), {impl.name()}")

                tp_correctness = impl(L1, L2, L3, self.correctness_batch_size)
                tp_bench = impl(L1, L2, L3, self.bench_batch_size)

                print("Started correctness check!")
                tp_correctness.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
                correctness, _ = tp_correctness.test_correctness(L1_in, L2_in, L3_out)
                if not correctness['pass']: print("Uh oh, correctness check failed!")
                print("Finished Correctness Check!")

                print("Started benchmark!")
                benchmark = tp_bench.benchmark(self.num_warmup, self.num_iter, self.bench_batch_size, prng_seed=self.prng_seed) 
                print("Completed benchmark!")
  
                if(False):
                    tp_correctness = impl(L1, L2, L3, self.correctness_batch_size)
                    print("Start GPU Array Benchmark")

                    gpu_L1_in = cp.cuda.runtime.malloc(L1_in.size)
                    cp.cuda.runtime.memcpy(gpu_L1_in, L1_in.ctypes.data, L1_in.nbytes, cp.cuda.runtime.memcpyHostToDevice)
                    print("one things copied")
                    
                    gpu_L2_in = cp.cuda.runtime.malloc(L2_in.size)
                    cp.cuda.runtime.memcpy(gpu_L2_in, L2_in.ctypes.data, L2_in.nbytes, cp.cuda.runtime.memcpyHostToDevice)
                    print("two things copied")

                    temp = np.zeros((self.correctness_batch_size, L3.mult(0), 2 * L3.type(0) + 1), dtype=np.float32)
                    gpu_L3_out = cp.cuda.runtime.malloc(temp.size)
                    cp.cuda.runtime.memcpy(gpu_L3_out, temp.ctypes.data, temp.nbytes, cp.cuda.runtime.memcpyHostToDevice)
                    print("three things copied")
                    
                    print(f"batch_size: {self.correctness_batch_size}, type: {type(self.correctness_batch_size)}")
                    tp_correctness.exec_tensor_product(self.correctness_batch_size, gpu_L1_in, gpu_L2_in, gpu_L3_out)
                    print('did da calc')
                    
                    cp.cuda.runtime.memcpy(temp.ctypes.data, gpu_L3_out, temp.nbytes, MEMCPY_DEVICE_TO_HOST)

                    print('got da answer')
                    correctness, _ = tp_correctness.test_correctness(L1_in, L2_in, gpu_L3_out)
                    print("gpu correctness")
                    print(correctness)

                    cp.cuda.runtime.free(gpu_L1_in)
                    cp.cuda.runtime.free(gpu_L2_in)
                    cp.cuda.runtime.free(gpu_L3_out)

                    print("Complete GPU Array Benchmark")



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
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.mult(0), 2 * L1.type(0) + 1)), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.mult(0), 2 * L2.type(0) + 1)), dtype=np.float32) 
    L3_out = np.zeros((batch_size, L3.mult(0), 2 * L3.type(0) + 1), dtype=np.float32)

    tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out)
    _ , ground_truth = tp.test_correctness(L1_in, L2_in, L3_out)

    print("Impl L3_out")
    print(L3_out)
    print("Ground Truth")
    print(ground_truth)
    print("diff")
    print(L3_out - ground_truth)

if __name__=='__main__':
    bench_suite = TestBenchmarkSuite()
    bench_suite.run([
        ThreadTensorProduct, 
        # GemmTensorProduct,
        ])
    # debug(ThreadTensorProduct, ((2, 4), (1, 3), (2, 5)))
