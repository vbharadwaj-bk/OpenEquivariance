import json, os, time, pickle

from build.kernel_wrapper import *
from src.implementations.AtomicConv import *

from src.benchmark.logging_utils import *
logger = getLogger()

import numpy as np
import numpy.linalg as la
import os

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
            return result["coords"], result["row"], result["col"]

    pickle_files = [f[:-7] for f in os.listdir("data/molecular_structures") if f.endswith(".pickle")]
    for candidate in pickle_files:
        if name == candidate:
            logger.info(f"Loading {name} from pickle...")
            coords, rows, cols = load_pickle(name) 
            logger.info(f"Graph {name} loaded.")

    if name == "debug":
        coords = np.array([[0.3, 0.4, 0.5], [0.3, 0.2, 0.1], [0.5, 0.4, 0.6]], dtype=np.float32)
        rows = np.array([0, 0, 1, 2, 2, 2], dtype=np.uint32)
        cols = np.array([0, 2, 2, 0, 1, 2], dtype=np.uint32)

    if coords is None or rows is None or cols is None:
        logger.critical(f"{bcolors.FAIL}Could not find graph with name {name}{bcolors.ENDC}")
        exit(1)

    return CoordGraph(coords, rows, cols)

def debug(conv_impl, rep_config, graph_name):
    logger.info("Starting debugging routine...")
    io_reps = config_to_rep_triple(rep_config)
    conv = conv_impl(io_reps)
    graph = load_graph(graph_name)

    rng = np.random.default_rng(12345)
    L1, L2, L3 = io_reps.L1, io_reps.L2, io_reps.L3
    L1_in  = np.array(rng.uniform(size=(graph.node_count, L1.get_rep_length())), dtype=np.float32) 
    L2_in  = np.array(rng.uniform(size=(graph.nnz, L2.get_rep_length())), dtype=np.float32) 
    L3_out = np.zeros((graph.node_count, L3.get_rep_length() ), dtype=np.float32)

    conv.exec_conv_cpu( L1_in, L2_in, L3_out, graph, disable_tensor_op=True)

    #conv.exec_conv_cpu(L1_in, L2_in, L3_out, graph, no_tensor_op=True)
    _ , ground_truth = conv.test_correctness_no_op(L1_in, L2_in, L3_out, graph)

    #print(L3_out) 
    #print(ground_truth) 
    #print(L3_out - ground_truth)
    print(la.norm((L3_out-ground_truth).flatten(), ord=np.inf))

if __name__=='__main__':
    rep_config = ("32x5e", "1x3e", "32x5e")
    debug(AtomicConv, rep_config, "debug")


