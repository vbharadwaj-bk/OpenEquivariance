import json, os, time, pickle, pathlib
import numpy as np
import numpy.linalg as la
import os

from build.kernel_wrapper import *
from src.implementations.AtomicConv import *
from src.implementations.SMConv import *

from src.benchmark.logging_utils import *
logger = getLogger()

def config_to_rep_triple(config):
    reps = None 
    if isinstance(config[0], tuple):
        reps = [Representation(config[i][0], config[i][1]) for i in range(3)]
    elif isinstance(config[0], str):
        reps = [Representation(config[i]) for i in range(3)] 
    return RepTriple(reps[0], reps[1], reps[2])

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
            logger.info(f"Graph {name} loaded.")

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

    def run(self, tp_implementations, correctness=True):        
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        graph = self.graph

        rep_sets = [config_to_rep_triple(config) for config in self.configs] 
        metadata = {
            "configs": [reps.to_string() for reps in rep_sets], 
            "implementations": [impl.name() for impl in tp_implementations],
            "graph": graph.name
        }
        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 

        for io_reps in rep_sets: 
            rng = np.random.default_rng(self.prng_seed)

            L1, L2, L3 = io_reps.L1, io_reps.L2, io_reps.L3

            L1_in  = np.array(rng.uniform(size=(graph.node_count, L1.get_rep_length())), dtype=np.float32) 
            L2_in  = np.array(rng.uniform(size=(graph.node_count, L2.get_rep_length())), dtype=np.float32) 
            L3_out = np.zeros((graph.node_count, L3.get_rep_length()), dtype=np.float32)

            for impl in tp_implementations:
                tc_name = f"{io_reps.to_string()}, {impl.name()}"
                logger.info(f'Starting {tc_name}, graph {graph.name}')

                conv = impl(io_reps)

                if correctness:
                    if self.disable_tensor_op:
                        conv.exec_conv_cpu( L1_in, L2_in, L3_out, self.graph, self.disable_tensor_op)
                        correctness, _ = conv.test_correctness_no_op(L1_in, L2_in, L3_out, self.graph)
                    else:
                        raise NotImplementedError("No correctness check implemented including tensor operation!")

                benchmark = conv.benchmark(self.num_warmup, 
                            self.num_iter, self.graph, self.disable_tensor_op, prng_seed=12345)

                rnames= [rep.to_string().replace(' ', '') for rep in [L1, L2, L3]]
                result = {
                    "config": rnames,
                    "graph": graph.name,
                    "name": impl.name(),
                    "correctness": correctness,
                    "benchmark": benchmark
                }
         
                fname = pathlib.Path(f"{output_folder}/{rnames[0]}_{rnames[1]}_{rnames[2]}_{impl.name()}.json")

                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f'Finished {tc_name}, graph {graph.name}')

def debug(conv_impl, rep_config, graph):
    logger.info("Starting debugging routine...")
    io_reps = config_to_rep_triple(rep_config)
    conv = conv_impl(io_reps)

    rng = np.random.default_rng(12345)
    L1, L2, L3 = io_reps.L1, io_reps.L2, io_reps.L3
    L1_in  = np.array(rng.uniform(size=(graph.node_count, L1.get_rep_length())), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(graph.nnz, L2.get_rep_length())), dtype=np.float32) 
    L3_out = np.zeros((graph.node_count, L3.get_rep_length() ), dtype=np.float32)

    conv.exec_conv_cpu( L1_in, L2_in, L3_out, graph, disable_tensor_op=True)
    _ , ground_truth = conv.test_correctness_no_op(L1_in, L2_in, L3_out, graph)

    #print(L3_out) 
    #print(ground_truth) 
    print(L3_out - ground_truth)
    print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))

if __name__=='__main__':
    graph = load_graph("covid_spike_radius3.5")
    rep_config = ("32x5e", "1x3e", "32x5e")

    bench = ConvBenchmarkSuite(
        [rep_config], graph,
        disable_tensor_op=True
    )
    bench.run([AtomicConv, SMConv]) 

    #debug(AtomicConv, rep_config, graph) 


