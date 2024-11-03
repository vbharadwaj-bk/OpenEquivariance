import json, os, time, pathlib 

from src.benchmark.logging_utils import *
from build.kernel_wrapper import *
from src.implementations.LoopUnrollTP import *
from src.implementations.NumpyTensorProduct import *
from src.implementations.e3nn_lite import *

import numpy as np
import numpy.linalg as la

logger = getLogger()

def mace_conf(irreps1, irreps2, lmax):
    '''
    Modified from mace/mace/modules/irreps_tools.py.
    '''
    trainable = True
    irreps1 = Irreps(irreps1)
    irreps2 = Irreps(irreps2)

    # Collect possible irreps and their instructions
    irreps_out_list = [] 
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out.l <= lmax: 
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    irreps_out = Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])
    result = TPProblem(irreps1, irreps2, irreps_out, instructions,
        internal_weights=False,
        shared_weights=False)
    result.metadata = f"mace_conf_{irreps1}__{irreps2}_{lmax}"
    result.metadata = result.metadata.replace(' ', '')
    return result

def single_inst_conf(irreps1, irreps2, irreps_out, mode, trainable):
    irreps1 = Irreps(irreps1)
    irreps2 = Irreps(irreps2)
    irreps_out = Irreps(irreps_out)
    instructions = [(0, 0, 0, mode, trainable)]

    result = TPProblem(irreps1, irreps2, irreps_out, instructions,
        internal_weights=False,
        shared_weights=False)
    result.metadata = f"single_inst_conf_{irreps1}__{irreps2}__{irreps_out}"
    result.metadata = result.metadata.replace(' ', '')
    return result 

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

    def run(self, tp_implementations, direction, correctness=True):        
        assert(direction == "forward" or direction == "backward")

        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        metadata = {
            "configs": [config.metadata for config in self.configs], 
            "implementations": [impl.name() for impl in tp_implementations]
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        for config in self.configs:
            L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out 
            rng = np.random.default_rng(self.prng_seed)
            L1_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L1.dim)), dtype=np.float32) 
            L2_in  = np.array(rng.uniform(size=(self.correctness_batch_size, L2.dim)), dtype=np.float32)

            # Assumes weights are not shared 
            weights = np.array(rng.uniform(size=(self.correctness_batch_size, config.weight_numel)), dtype=np.float32) 
            L3_out = np.zeros((self.correctness_batch_size, L3.dim), dtype=np.float32)
            for impl in tp_implementations:
                tc_name = f"{config.metadata}, {impl.name()}, {direction}"
                logger.info(f'Starting {tc_name}.')

                if correctness and direction == "forward":
                    tp_correctness = impl(config)
                    tp_correctness.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)
                    correctness, _ = tp_correctness.test_correctness(L1_in, L2_in, weights, L3_out,
                        reference_implementation=NumpyTensorProduct)

                tp_bench = impl(config)
                benchmark = tp_bench.benchmark(self.num_warmup, self.num_iter, self.bench_batch_size, direction, prng_seed=self.prng_seed) 
                result = {
                    "config": config.metadata,
                    "direction": direction, 
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark
                }
         
                fname = pathlib.Path(f"{output_folder}/{config.metadata}.json")

                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f'Finished {tc_name}.')

def debug(tp_impl, config, direction="forward"): 
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out 
    batch_size = 1

    tp = tp_impl(config)

    rng = np.random.default_rng(12345)
    L1_in  = np.array(rng.uniform(size=(batch_size, L1.dim)), dtype=np.float32)
    L2_in  = np.array(rng.uniform(size=(batch_size, L2.dim)), dtype=np.float32)
    weights = np.array(rng.uniform(size=(batch_size, config.weight_numel)), dtype=np.float32) 

    L3_out = np.zeros((batch_size, L3.dim), dtype=np.float32)

    if direction == "forward":
        tp.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)
        _, ground_truth = tp.test_correctness(L1_in, L2_in, weights, L3_out)
        print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))
        print(L3_out / ground_truth)
    elif direction == "backward":
        L3_grad = L3_out
        L3_grad[:] = rng.uniform(size=(batch_size, L3.dim)) 
        weights = np.array(rng.uniform(size=(batch_size, config.weight_numel)), dtype=np.float32) # Assumes no shared weights
        L1_grad, L2_grad, weights_grad = tp.backward_cpu(L1_in, L2_in, L3_grad, weights)
        print(L1_grad)
        print(L2_grad)
        print(weights_grad)
    else:
        assert(False)

if __name__=='__main__':
    tests = [
        single_inst_conf("32x5e", "1x5e", "32x3e", "uvu", True),
        #("32x3e + 32x2e", "1x0e + 1x1e", 3), # Last value is Lmax
        #("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3), 
        #("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]

    bench_suite = TestBenchmarkSuite(tests, bench_batch_size=1000000)
    bench_suite.run([LoopUnrollTP], direction="forward")

    #debug(LoopUnrollTP, tests[0], direction="forward")