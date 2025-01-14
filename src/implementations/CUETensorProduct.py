import numpy as np
import tempfile, json

from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import *
from src.benchmark.logging_utils import getLogger
from src.benchmark.tpp_creation_utils import *
from build.kernel_wrapper import *

logger = getLogger()

class CUETensorProduct(TensorProduct):
    def __init__(self, config : TPProblem, torch_op=True):
        super().__init__(config, torch_op=torch_op)

        global torch
        import torch
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        import e3nn.o3 as o3

        supported_tpp_types = [
            ChannelwiseTPP,
            FullyConnectedTPProblem,
            SingleInstruction
        ]

        assert(config.irrep_dtype == config.weight_dtype)

        np_to_torch_dtype = {
            np.float32: torch.float32,
            np.float64: torch.float64
        }

        class O3_e3nn(cue.O3):
            def __mul__(  # pylint: disable=no-self-argument
                rep1: "O3_e3nn", rep2: "O3_e3nn"
            ) -> Iterator["O3_e3nn"]:
                return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

            @classmethod
            def clebsch_gordan(
                cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
            ) -> np.ndarray:
                rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

                if rep1.p * rep2.p == rep3.p:
                    return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                        rep3.dim
                    )
                return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

            def __lt__(  # pylint: disable=no-self-argument
                rep1: "O3_e3nn", rep2: "O3_e3nn"
            ) -> bool:
                rep2 = rep1._from(rep2)
                return (rep1.l, rep1.p) < (rep2.l, rep2.p)

            @classmethod
            def iterator(cls) -> Iterator["O3_e3nn"]:
                for l in itertools.count(0):
                    yield O3_e3nn(l=l, p=1 * (-1) ** l)
                    yield O3_e3nn(l=l, p=-1 * (-1) ** l)

        self.cue_tp = None
        torch_dtype = np_to_torch_dtype[config.irrep_dtype]

        assert(any([isinstance(config, supported_ttp_type)] for supported_ttp_type in supported_tpp_types))
        if isinstance(config, ChannelwiseTPP) or isinstance(config, SingleInstruction):
            self.cue_tp = cuet.ChannelWiseTensorProduct(
                cue.Irreps(O3_e3nn, str(config.irreps_in1)),
                cue.Irreps(O3_e3nn, str(config.irreps_in2)),
                cue.Irreps(O3_e3nn, str(config.irreps_out)),
                layout=cue.ir_mul,
                shared_weights=config.shared_weights,
                internal_weights=config.internal_weights,
                dtype=torch_dtype,
                math_dtype=torch_dtype
            )
            self.cue_tp.to('cuda')
            self.forward = self.cue_tp.__call__
        
        if isinstance(config, FullyConnectedTPProblem):
            e = cue.descriptors.fully_connected_tensor_product(
                cue.Irreps("O3", str(config.irreps_in1)),
                cue.Irreps("O3", str(config.irreps_in2)),
                cue.Irreps("O3", str(config.irreps_out)),
            )

            assert(config.weight_numel == e.inputs[0].irreps.dim)
            self.cue_tp = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul,
                    math_dtype=np_to_torch_dtype[config.irrep_dtype]) 

            self.cue_tp.to('cuda')
            self.forward = lambda x, y, W: self.cue_tp(W, x, y)

    def analyze_trace(self, trace_file):
        trace = None
        with open(trace_file, "r") as f:
            trace = json.load(f)

        tp_time = 0.0
        total = 0.0

        for event in trace["traceEvents"]:
            if "args" in event and "stream" in event["args"]:
                event_time_ms = event["dur"] / 1000
                total += event_time_ms 

                if "TensorProductUniform1dKernel" in event["name"]:
                    tp_time += event_time_ms 

        logger.info("Total time: %f ms, TP time: %f ms", total, tp_time)
        return tp_time 

    def benchmark_forward(
            self, 
            num_warmup : int, 
            num_iter : int, 
            L1_in : np.ndarray, 
            L2_in : np.ndarray, 
            L3_buffer : np.ndarray, 
            weights : np.ndarray) -> np.ndarray:
        '''
        When we don't want to include torch overhead, we use the Pytorch profiler
        to extract the device time that the kernel takes.
        '''
        if self.torch_op:
            return super().benchmark_forward(num_warmup, num_iter, L1_in, L2_in, L3_buffer, weights)
        else:
            from torch.profiler import profile, record_function, ProfilerActivity
            time_millis = np.zeros(num_iter, dtype=np.float32)
            torch_L1_in = torch.tensor(L1_in).to(device='cuda').detach()
            torch_L2_in = torch.tensor(L2_in).to(device='cuda').detach()
            torch_weights = torch.tensor(weights).to(device='cuda').detach()

            timer = GPUTimer()

            for i in range(num_warmup):
                torch_L3_out = self.forward(torch_L1_in, torch_L2_in, torch_weights) 

            trace_file = tempfile.NamedTemporaryFile().name

            for i in range(num_iter):
                timer.clear_L2_cache()
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("cue_forward"):
                        torch_L3_out = self.forward(torch_L1_in, torch_L2_in, torch_weights) 

                prof.export_chrome_trace(trace_file)
                time_millis[i] = self.analyze_trace(trace_file)

            return time_millis
 
    @staticmethod
    def name():
        return "CUETensorProduct"
