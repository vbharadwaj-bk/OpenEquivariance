import json, os, time, pickle, pathlib
import numpy as np
import numpy.linalg as la
import os

import openequivariance
from openequivariance.kernel_wrapper import *
from openequivariance.benchmark.tpp_creation_utils import *
from openequivariance.implementations.convolution.LoopUnrollConv import *
from openequivariance.implementations.convolution.CUEConv import *

from openequivariance.benchmark.logging_utils import *
from openequivariance.benchmark.ConvBenchmarkSuite import *
logger = getLogger()

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
    clean_benchmark()
    exit(1)

    #graph = load_graph("debug")
    #graph = load_graph("covid_spike_radius3.0")
    #graph = load_graph("carbon_lattice_radius6.0")
    #config= SingleInstruction("32x5e", "1x3e", "32x5e", "uvu", True)

    configs = [
        #SingleInstruction("1x2e", "1x2e", "1x2e", "uvu", True),
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
            [   LoopUnrollConvScatterSum, 
                #CUEConv,
                #LoopUnrollConvDeterministic, 
                #LoopUnrollConvAtomic
                ], 
            direction="forward", 
            correctness=False,
            double_backward_correctness=False,
            benchmark=True)

    #debug(LoopUnrollConv, configs[0], graph, direction="backward")