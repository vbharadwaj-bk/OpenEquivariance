import json, os, time, pathlib

import numpy as np

from src.benchmark.logging_utils import *
from build.kernel_wrapper import *
from src.implementations.e3nn_lite import *

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
    result.metadata = f"mace_conf_{irreps1}__{irreps2}__{lmax}"
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

    def run(self, tp_implementations, direction, reference_impl):        
        assert(direction == "forward" or direction == "backward")

        correctness = None

        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        metadata = {
            "test_name": "Batch TP",
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
                tc_name = f"{config.metadata}, {bcolors.OKCYAN}{impl.name()}{bcolors.ENDC}, {direction}"
                logger.info(f'Starting {tc_name}.')

                if reference_impl is not None and direction == "forward":
                    tp_correctness = impl(config)
                    tp_correctness.exec_tensor_product_cpu(L1_in, L2_in, L3_out, weights)
                    correctness, _ = tp_correctness.test_correctness(L1_in, L2_in, weights, L3_out,
                        reference_implementation=reference_impl)
                else:
                    logger.warning(f"{bcolors.WARNING}Correctness check skipped.{bcolors.ENDC}")

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