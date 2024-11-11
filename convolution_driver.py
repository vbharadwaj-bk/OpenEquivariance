import json, os, time, pickle, pathlib
import numpy as np
import numpy.linalg as la
import os

from build.kernel_wrapper import *
from src.implementations.LoopUnrollConv import *
from src.implementations.NumpyConv import *

from src.benchmark.TestBenchmarkSuite import mace_conf, single_inst_conf  

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
        rows = np.array([0, 0, 1, 2, 2, 2], dtype=np.uint32)
        cols = np.array([0, 2, 2, 0, 1, 2], dtype=np.uint32)
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
        prng_seed = 12345
    ):
        self.configs = configs
        self.graph = graph
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.disable_tensor_op = disable_tensor_op
        self.prng_seed = 12345

    def run(self, tp_implementations, direction, correctness=True):        
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        graph = self.graph

        metadata = {
            "test_name": "Convolution",
            "configs": [config.metadata for config in self.configs], 
            "implementations": [impl.name() for impl in tp_implementations],
            "graph": graph.name
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        for config in self.configs: 
            L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
            rng = np.random.default_rng(self.prng_seed)

            L1_in  = np.array(rng.uniform(size=(graph.node_count, L1.dim)), dtype=np.float32)
            L2_in  = np.array(rng.uniform(size=(graph.nnz, L2.dim)), dtype=np.float32)
            weights = np.array(rng.uniform(size=(graph.nnz, config.weight_numel)), dtype=np.float32)
            L3_out = np.zeros((graph.node_count, L3.dim), dtype=np.float32)

            for impl in tp_implementations:
                tc_name = f"{config.metadata}, {impl.name()}"
                logger.info(f'Starting {tc_name}, graph {graph.name}, {direction}')
                conv = impl(config)

                if correctness and direction == "forward":
                    conv.exec_conv_cpu( L1_in, L2_in, weights, L3_out, self.graph, self.disable_tensor_op)
                    correctness, _ = conv.test_correctness(L1_in, L2_in, weights, L3_out, self.graph,
                            conv_reference_impl=NumpyConv, disable_tensor_op=self.disable_tensor_op)

                benchmark = conv.benchmark(self.num_warmup,
                            self.num_iter, self.graph, self.disable_tensor_op, direction, prng_seed=12345)

                result = {
                    "config": config.metadata,
                    "graph": graph.name,
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark
                }
         
                fname = pathlib.Path(f"{output_folder}/{config.metadata}_{impl.name()}_{graph.name}.json")

                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f'Finished {tc_name}, graph {graph.name}')

def debug(conv_impl, config, graph, direction, disable_tensor_op=False):
    logger.info("Starting debugging routine...")
    conv = conv_impl(config)

    rng = np.random.default_rng(12345)
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
    L1_in  = np.array(rng.uniform(size=(graph.node_count, L1.dim)), dtype=np.float32)
    L2_in  = np.array(rng.uniform(size=(graph.nnz, L2.dim)), dtype=np.float32)
    weights = np.array(rng.uniform(size=(graph.nnz, config.weight_numel)), dtype=np.float32)
    L3_out = np.zeros((graph.node_count, L3.dim), dtype=np.float32)

    if direction == "forward":
        conv.exec_conv_cpu( L1_in, L2_in, weights, L3_out, graph, disable_tensor_op=disable_tensor_op)
        correctness, ground_truth = conv.test_correctness(L1_in, L2_in, weights, L3_out, graph, 
                conv_reference_impl=NumpyConv, disable_tensor_op=disable_tensor_op)

        print((L3_out - ground_truth)[:2, 0])
        print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))
    elif direction == "backward":
        L3_grad = L3_out
        L3_grad[:] = rng.uniform(size=(graph.node_count, L3.dim)) 

        L1_grad, L2_grad, weights_grad = conv.backward_cpu( 
            L1_in, L2_in, weights, L3_grad,
            graph, False)

if __name__=='__main__':
    graph = load_graph("covid_spike_radius3.5")
    config= single_inst_conf("32x5e", "1x3e", "32x5e", "uvu", True)

    configs = [
        single_inst_conf("32x5e", "1x3e", "32x5e", "uvu", True),
        single_inst_conf("32x5e", "1x5e", "32x3e", "uvu", True),
        mace_conf("32x3e + 32x2e", "1x0e + 1x1e", 3),
        mace_conf("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3),
        mace_conf("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]

    cut_size = len(graph.rows)
    graph.rows = graph.rows[:cut_size]
    graph.cols = graph.cols[:cut_size]
    graph.nnz = cut_size

    bench = ConvBenchmarkSuite(
        configs, graph,
        disable_tensor_op=False
    )
    bench.run([LoopUnrollConv], direction="backward", correctness=False)

    #debug(LoopUnrollConv, configs[0], graph, direction="backward", disable_tensor_op=True)