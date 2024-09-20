import json, os, time, pathlib 

from build.kernel_wrapper import *
from src.implementations.Convolution import *

from src.benchmark.logging_utils import *
logger = getLogger()

import numpy as np
import numpy.linalg as la

def config_to_rep_triple(config):
    reps = None 
    if isinstance(config[0], tuple):
        reps = [Representation(config[i][0], config[i][1]) for i in range(3)]
    elif isinstance(config[0], str):
        reps = [Representation(config[i]) for i in range(3)] 
    return RepTriple(reps[0], reps[1], reps[2])

if __name__=='__main__':
    config = ("32x5e", "1x5e", "32x3e")
    conv = Convolution(config_to_rep_triple(config))

