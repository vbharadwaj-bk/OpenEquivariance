from src.implementations.Convolution import *
from build.kernel_wrapper import *

class AtomicConv(Convolution):
    def __init__(self, io_reps: RepTriple):
        super().__init__(io_reps)
        self.internal = AtomicConvImpl(io_reps) 

    @staticmethod
    def name():
        return "AtomicConvolution"