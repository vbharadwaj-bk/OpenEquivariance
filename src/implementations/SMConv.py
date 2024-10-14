from src.implementations.Convolution import *
from build.kernel_wrapper import *

class SMConv(Convolution):
    def __init__(self, io_reps: RepTriple):
        super().__init__(io_reps)
        self.internal = SMConvImpl(io_reps) 

    @staticmethod
    def name():
        return "SharedMemoryConvolution"