import json, os, time, pickle, pathlib
import numpy as np
import numpy.linalg as la
import os

from build.kernel_wrapper import *
from src.benchmark.tpp_creation_utils import *
from src.implementations.LoopUnrollConv import *
from src.implementations.NumpyConv import *
from src.implementations.E3NNConv import *

from src.benchmark.logging_utils import *
logger = getLogger()

def load_graph(name):
    coords, rows, cols = None, None, None

    def load_pickle(name):
        with open(f"data/molecular_structures/{name}.pickle", 'rb') as f:
            result = pickle.load(f)
            return result["coords"], result["row"], result["col"], name

    pickle_files = [f[:-7] for f in os.listdir("data/molecular_structures") if f.endswith(".pickle")]
    for candidate in pickle_files:
        if name == candidate:
            logger.info(f"Loading {name} from pickle...")
            coords, rows, cols, name = load_pickle(name) 
            logger.info(f"Graph {name} loaded with {len(coords)} nodes and {len(rows)} edges.")

    if name == "debug":
        coords = np.array([[0.3, 0.4, 0.5], [0.3, 0.2, 0.1], [0.5, 0.4, 0.6]], dtype=np.float32)
        rows = np.array([0, 1, 1, 2, 2, 2], dtype=np.uint32)
        cols = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)

        #coords = coords[:1]
        #rows = rows[:1]
        #cols = cols[:1] 

        name = "debug" 

    if coords is None or rows is None or cols is None:
        logger.critical(f"{bcolors.FAIL}Could not find graph with name {name}{bcolors.ENDC}")
        exit(1)

    return CoordGraph(coords, rows, cols, name)


class ConvBenchmarkSuite:
    def __init__(self, configs, graph,
        num_warmup = 10,
        num_iter = 30,
        disable_tensor_op=False,
        reference_impl=E3NNConv,
        prng_seed = 12345
    ):
        self.configs = configs
        self.graph = graph
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.disable_tensor_op = disable_tensor_op
        self.reference_impl = reference_impl
        self.prng_seed = 12345
        self.correctness_threshold = 1e-5

    def run(self, tp_implementations, direction, correctness=True):        
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        graph = self.graph

        metadata = {
            "test_name": "Convolution",
            "configs": [str(config) for config in self.configs], 
            "implementations": [impl.name() for impl in tp_implementations],
            "graph": graph.name
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        exp_count = 0

        for config in self.configs: 
            L1_in, L2_in, weights, L3_out = get_random_buffers_forward_conv(config, graph.node_count, graph.nnz, self.prng_seed)

            for impl in tp_implementations:
                tc_name = f"{config}, {impl.name()}"
                logger.info(f'Starting {tc_name}, graph {graph.name}, {direction}')
                conv = impl(config)
                benchmark = None

                if direction == "forward":
                    if correctness:
                        correctness = conv.test_correctness_forward(self.graph, 
                                thresh=self.correctness_threshold, 
                                prng_seed=self.prng_seed, 
                                reference_implementation=self.reference_impl)

                    benchmark = conv.benchmark_forward(self.num_warmup,
                                self.num_iter, self.graph, self.disable_tensor_op, prng_seed=12345)


                if direction == "backward":
                    if correctness:
                        correctness = conv.test_correctness_backward(self.graph, 
                                thresh=self.correctness_threshold, 
                                prng_seed=self.prng_seed, 
                                reference_implementation=self.reference_impl)

                    benchmark = conv.benchmark_backward(self.num_warmup,
                                self.num_iter, self.graph, self.disable_tensor_op, prng_seed=12345)

                result = {
                    "config": str(config),
                    "direction": direction,
                    "graph": graph.name,
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark 
                }
         
                fname = pathlib.Path(f"{output_folder}/{exp_count}_{impl.name()}_{graph.name}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)
                exp_count += 1

                logger.info(f'Finished {tc_name}, graph {graph.name}')

if __name__=='__main__':
    #graph = load_graph("debug")
    graph = load_graph("covid_spike_radius3.5")
    #config= SingleInstruction("32x5e", "1x3e", "32x5e", "uvu", True)

    configs = [
        SingleInstruction("32x5e", "1x3e", "32x5e", "uvu", True),
        #ChannelwiseTPP("128x2e + 128x1o + 128x0e", "1x0e + 1x1e", 3),
        #SingleInstruction("32x5e", "1x5e", "32x3e", "uvu", True),
        #ChannelwiseTPP("32x3e + 32x2e", "1x0e + 1x1e", 3),
        #ChannelwiseTPP("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3),
        #ChannelwiseTPP("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]

    for config in configs:
        config.irrep_dtype = np.float64
        config.weight_dtype = np.float64

    cut_size = len(graph.rows)
    graph.rows = graph.rows[:cut_size]
    graph.cols = graph.cols[:cut_size]
    graph.nnz = cut_size

    bench = ConvBenchmarkSuite(
        configs, graph,
        disable_tensor_op=False)
    bench.run([LoopUnrollConv], direction="forward", correctness=True)

    #debug(LoopUnrollConv, configs[0], graph, direction="backward", disable_tensor_op=True)