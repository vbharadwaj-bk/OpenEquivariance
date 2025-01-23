import json, os, time, pickle, pathlib
import numpy as np
import numpy.linalg as la
import os

from openequivariance import package_root
from openequivariance.benchmark.logging_utils import *
from openequivariance.implementations.convolution.Convolution import *
logger = getLogger()

def load_graph(name):
    coords, rows, cols = None, None, None

    def load_pickle(name):
        with open(f"../data/molecular_structures/{name}.pickle", 'rb') as f:
            result = pickle.load(f)
            return result["coords"], result["row"], result["col"], name

    pickle_files = [f[:-7] for f in os.listdir("../data/molecular_structures") if f.endswith(".pickle")]
    for candidate in pickle_files:
        if name == candidate:
            logger.info(f"Loading {name} from pickle...")
            coords, rows, cols, name = load_pickle(name) 
            logger.info(f"Graph {name} loaded with {len(coords)} nodes and {len(rows)} edges.")

    if name == "debug":
        coords = np.array([[0.3, 0.4, 0.5], [0.3, 0.2, 0.1], [0.5, 0.4, 0.6]], dtype=np.float32)
        rows = np.array([0, 1, 1, 2], dtype=np.uint32)
        cols = np.array([1, 0, 2, 1], dtype=np.uint32)

        coords = coords[:4]
        rows = rows[:4]
        cols = cols[:4] 

        name = "debug" 

    if coords is None or rows is None or cols is None:
        logger.critical(f"{bcolors.FAIL}Could not find graph with name {name}{bcolors.ENDC}")
        exit(1)

    return CoordGraph(coords, rows.astype(np.int64), cols.astype(np.int64), name)

class ConvBenchmarkSuite:
    def __init__(self, configs, 
        num_warmup = 10,
        num_iter = 30,
        reference_impl=None,
        torch_op=True,
        prng_seed = 12345
    ):
        self.configs = configs
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.reference_impl = reference_impl
        self.prng_seed = 12345
        self.correctness_threshold = 1e-5
        self.torch_op = torch_op

        millis_since_epoch = round(time.time() * 1000)
        self.output_folder = pathlib.Path(f'{package_root}/outputs/{millis_since_epoch}')
        self.output_folder.mkdir(parents=True)
        self.exp_count = 0

    def run(self, graph, implementations, direction, correctness=True, double_backward_correctness=False, benchmark=True):        
        metadata = {
            "test_name": "Convolution",
            "configs": [str(config) for config in self.configs], 
            "implementations": [impl.name() for impl in implementations],
            "graph": graph.name
        }
        if self.exp_count == 0:
            with open(os.path.join(self.output_folder,'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2) 

        for config in self.configs: 
            L1_in, L2_in, weights, L3_out = get_random_buffers_forward_conv(config, graph.node_count, graph.nnz, self.prng_seed)

            for impl in implementations:
                tc_name = f"{config}, {impl.name()}"
                logger.info(f'Starting {tc_name}, graph {graph.name}, {direction}')
                conv = impl(config, torch_op=self.torch_op)

                if double_backward_correctness:
                    double_backward_correctness = conv.test_correctness_double_backward(self.graph, 
                            thresh=self.correctness_threshold, 
                            prng_seed=self.prng_seed, 
                            reference_implementation=self.reference_impl)

                if direction == "forward":
                    if correctness:
                        correctness = conv.test_correctness_forward(graph, 
                                thresh=self.correctness_threshold, 
                                prng_seed=self.prng_seed, 
                                reference_implementation=self.reference_impl)

                    if benchmark:
                        benchmark = conv.benchmark_forward(self.num_warmup,
                                    self.num_iter, graph, prng_seed=12345)


                if direction == "backward":
                    if correctness:
                        correctness = conv.test_correctness_backward(graph, 
                                thresh=self.correctness_threshold, 
                                prng_seed=self.prng_seed, 
                                reference_implementation=self.reference_impl)

                    if benchmark:
                        benchmark = conv.benchmark_backward(self.num_warmup,
                                    self.num_iter, graph, prng_seed=12345)

                result = {
                    "config": str(config),
                    "irrep_dtype": str(config.irrep_dtype),
                    "weight_dtype": str(config.weight_dtype),
                    "torch_overhead_included": self.torch_op,
                    "direction": direction,
                    "graph": graph.name,
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark,
                    "double_backward_correctness": double_backward_correctness
                }
         
                fname = pathlib.Path(f"{self.output_folder}/{self.exp_count}_{impl.name()}_{graph.name}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)
                self.exp_count += 1

                logger.info(f'Finished {tc_name}, graph {graph.name}')
