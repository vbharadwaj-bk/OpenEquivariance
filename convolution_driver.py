import json, os, time, pickle, pathlib
import numpy as np
import numpy.linalg as la
import os

from src.extlib.kernel_wrapper import *
from src.benchmark.tpp_creation_utils import *
from src.implementations.convolution.LoopUnrollConv import *
from src.implementations.convolution.CUEConv import *

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
        cols = np.array([1, 0, 2, 0, 1, 2], dtype=np.uint32)

        coords = coords[:2]
        rows = rows[:2]
        cols = cols[:2] 

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
        self.output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
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

def clean_benchmark():
    covid_spike = load_graph("covid_spike_radius3.0")
    dhfr = load_graph("1drf_radius6.0")
    carbon = load_graph("carbon_lattice_radius6.0")

    configs = [ ChannelwiseTPP("128x0e+128x1o+128x2e", 
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e+128x3o"),
                ChannelwiseTPP("128x0e+128x1o+128x2e", 
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e+128x3o"),
                ] # MACE-large 

    configs[1].irrep_dtype = np.float64
    configs[1].weight_dtype = np.float64

    bench = ConvBenchmarkSuite(configs, torch_op=True)

    implementations = [ LoopUnrollConvScatterSum, 
                        CUEConv,
                        LoopUnrollConvDeterministic, 
                        LoopUnrollConvAtomic
                        ]

    for graph in [covid_spike, dhfr, carbon]:
        for direction in ["forward", "backward"]:
            bench.run(
                    implementations = implementations,
                    graph = graph,
                    direction=direction, 
                    correctness=False,
                    double_backward_correctness=False,
                    benchmark=True)


if __name__=='__main__':
    #clean_benchmark()
    #exit(1)
    #graph = load_graph("debug")
    graph = load_graph("carbon_lattice_radius6.0")
    #config= SingleInstruction("32x5e", "1x3e", "32x5e", "uvu", True)

    configs = [
        #SingleInstruction("32x5e", "1x3e", "32x5e", "uvu", True),
        ChannelwiseTPP("128x0e+128x1o+128x2e", 
                "1x0e+1x1o+1x2e+1x3o",
                "128x0e+128x1o+128x2e+128x3o"),
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
        configs, torch_op=True)
    bench.run( graph,
            [   #LoopUnrollConvScatterSum, 
                #CUEConv,
                LoopUnrollConvDeterministic, 
                #LoopUnrollConvAtomic
                ], 
            direction="forward", 
            correctness=True,
            double_backward_correctness=False,
            benchmark=True)

    #debug(LoopUnrollConv, configs[0], graph, direction="backward")