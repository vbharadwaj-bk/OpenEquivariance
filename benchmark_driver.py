import numpy as np
import numpy.linalg as la

from src.benchmark.logging_utils import *

from src.implementations.LoopUnrollTP import LoopUnrollTP
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction

conv_problems = [  
    mace_conf("128x2e + 128x1o + 128x0e", "1x0e + 1x1e", 3)
]
