import pickle, pathlib
import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *

from src.benchmark.logging_utils import getLogger, bcolors 
logger = getLogger()

class Convolution:
    def __init__(self, io_reps: RepTriple):
        self.L1, self.L2, self.L3 = io_reps.L1, io_reps.L2, io_reps.L3
        self.internal = None


    @staticmethod
    def name():
        raise NotImplementedError()


