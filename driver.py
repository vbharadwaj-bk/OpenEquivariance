import json, os, time, pathlib 

from src.benchmark.logging_utils import *
from src.benchmark.e3nn_tp_utils import *
from build.kernel_wrapper import *
from src.implementations.GemmTP import *
from src.implementations.ThreadTP import *
from src.implementations.ShuffleReduceTP import *
from src.implementations.LoopUnrollTP import *
from src.implementations.MultiplicityOuterProductTP import *

import numpy as np
import numpy.linalg as la

import torch
import e3nn
from e3nn.o3 import Irrep, Irreps


logger = getLogger()

class TestBenchmarkSuite:
    def __init__(
        self,
        num_warmup = 10,
        num_iter = 30,
        correctness_batch_size = 10000,
        bench_batch_size = 10000000,
        prng_seed = 12345,
        correctness_threshold = 5e-7, # Above floating point interval machine epsilon 
    ):
        self.num_warmup = num_warmup
        self.num_iter = num_iter
        self.correctness_batch_size = correctness_batch_size 
        self.bench_batch_size = bench_batch_size 
        self.prng_seed = prng_seed
        self.correctness_threshold = correctness_threshold
    
    def validate_inputs(self, test_list):
        """
        Just does empty list and type checking to catch bad input 
        """
        assert len(test_list) != 0
        for config, implementation, direction, do_correctness, do_benchmark in test_list:
            assert isinstance(config, e3nn.o3.TensorProduct)
            assert isinstance(implementation, TensorProduct)
            assert direction in ["forward", "backward"]
            assert isinstance(do_correctness,bool)
            assert isinstance(do_benchmark, bool)
    
    def generate_metadata(self, test_list):
        """
        creates an (incomplete) summary of what was tested
        """
        config_names = set()
        implementation_names = set()
        directions = set()
        did_correctness = False
        did_benchmark = False
        for config, implementation, direction, do_correctness, do_benchmark in test_list:
            config_names += convenience_namer(config.irreps_in1, config.irreps_in2, config.ireps_out) 
            implementation_names += implementation.name()
            directions += direction
            did_correctness = did_correctness or do_correctness
            did_benchmark = did_benchmark or do_benchmark 

        metadata = {
                "configs": config_names,
                "implementations": implementation_names,
                "directions": directions,
                "did_correctness": did_correctness, 
                "did_benchmark":did_benchmark,
            }
        
        return metadata

    def check_similiarity(self, name : str,  to_check : np.ndarray,  ground_truth : np.ndarray):
        result = {}
        if to_check.shape != ground_truth.shape:
            result["shape_match"] = False
            result["diff_Linf_norm"] = np.inf
            result["pass"] = False
            logger.error(f"{bcolors.FAIL}Ground truth {name} shape does not match input! {to_check.shape=}, {ground_truth.shape=} {bcolors.ENDC}")
        else:
            result["shape_match"] = True 
            diff_Linf_norm = float(la.norm((ground_truth - to_check).flatten(), ord=np.inf))
            result["diff_Linf_norm"] = diff_Linf_norm 
            result["pass"] = bool(diff_Linf_norm < self.correctness_threshold) 

            if result["pass"]:
                logger.info(f"{bcolors.OKGREEN}{name} correctness check pass. {bcolors.ENDC}")
            else:
                logger.error(f"{bcolors.FAIL}{name} correctness check fail! {diff_Linf_norm=}, {self.correctness_threshold=} {bcolors.ENDC}")

        return result


    def test_correctness_forward(self, implementation : TensorProduct) -> dict:
        result = {
            "thresh": self.correctness_threshold, 
        }

        ### GENERATE INPUTS 
        in1, in2, weights, out = get_random_forward_supplies(implementation.e3nn_tp, self.correctness_batch_size, self.prng_seed)

        ### RUN GROUND TRUTH
        ground_truth_out = self.e3nn_tp(torch.Tensor(in1), torch.Tesnor(in2), torch.Tensor(weights)).numpy(force=True)

        ### RUN THE INTERNAL VERSION
        implementation.exec_tensor_product_cpu(
            in1, 
            in2, 
            out, 
            weights)
        
        ## CHECK SIMILARITY 
        for name, to_check, ground_truth in [
            ("output", out, ground_truth_out)
            ]:
            result[name] = self.check_similiarity(name, to_check, ground_truth)
        
        return result

    def test_correctness_backward(self, implementation : TensorProduct) -> dict:
        result = {
            "diff_Linf_norm": np.inf,
            "thresh": self.correctness_threshold, 
            "pass": False
        }

        ### GENERATE INPUTS
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_backward_supplies(
            implementation.e3nn_tp, 
            self.correctness_batch_size, 
            prng_seed=self.prng_seed
            )

        ### RUN THE GROUND TRUTH 
        torch_in1 = torch.Tensor(in1, requires_grad=True)
        torch_in2 = torch.Tensor(in2, requires_grad=True)
        torch_weights = torch.Tensor(weights, requires_grad=True)

        ground_truth_out = self.e3nn_tp.backward(torch_in1, torch_in2, torch_weights)

        ground_truth_out.backward(out_grad)

        ground_truth_weight_grad = torch_weights.grad.numpy(force=True)
        ground_truth_in1_grad = torch_in1.grad.numpy(force=True)
        ground_truth_in2_grad = torch_in2.grad.numpy(force=True)

        ### RUN THE INTERNAL VERSION 
        implementation.backward_cpu(
            L1_in=in1, 
            L1_grad=in1_grad,
            L2_in=in2, 
            L2_grad=in2_grad,
            weights=weights,
            weights_grad=weights_grad
            )
        
        ## CHECK OUTPUT SIMILARITY 
        for name, to_check, ground_truth in [
            ("weight_grad", weights_grad, ground_truth_weight_grad),
            ("in1_grad", in1_grad, ground_truth_in1_grad),
            ("in2_grad", in2_grad, ground_truth_in2_grad),
            ]:
            result[name] = self.check_similiarity(name, to_check, ground_truth)
        
        return result
    
    def calculate_performance_statistics(
            self, 
            implementation : TensorProduct, 
            total_flops : int, 
            total_streamed_memory : int, 
            time_millis : np.ndarray
            ) -> dict:
        result = {}

        throughputs_gflops = [float(x) for x in total_flops / (time_millis * 1e6)]
       
        bandwidth_gbps = [float(x) for x in total_streamed_memory / (time_millis * 1e6)]

        nnz = calculate_total_nnz(implementation)

        time_millis = [float(x) for x in time_millis]

        result += {
            "total_cg_nnz": nnz,
            "flops_per_tp": total_flops / self.bench_batch_size,
            "L1": implementation.L1.to_string(),
            "L2": implementation.L2.to_string(),
            "L3": implementation.L3.to_string(),

            "L1_rep_len": implementation.L1.get_rep_length(),
            "L2_rep_len": implementation.L2.get_rep_length(),
            "L3_rep_len": implementation.L3.get_rep_length(),

            "rep_dtype": "float", # If this changes, also need to modify the arithmetic intensity calculation
            "arithmetic_intensity (FLOPs / byte)": total_flops * total_streamed_memory, 

            "num_warmup": self.num_warmup,
            "num_iter": self.num_iter,
            "prng_seed": self.prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps": bandwidth_gbps,
        }

        logger.info(f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")

        return result                               

    def benchmark_forward(self, implementation : TensorProduct):
        '''
        This function sets up the necessary materials and calls the internal benchmarker
        '''
        result = {
            "tp_direction": "forward",
        }

        L1_in, L2_in, weights, L3_buffer = get_random_forward_supplies(self.bench_batch_size)

        logger.info("Initialized input / output data.")

        # BENCHMARK 
        time_millis = implementation.benchmark_internal_forward(
            num_warmup=self.num_warmup, 
            num_iter=self.num_iter,
            L1_in=L1_in,
            L2_in=L2_in,
            weights=weights,
            L3_buffer=L3_buffer
            )
        
        try:
            flops = implementation.calculate_forward_flops(batch_size=self.bench_batch_size)
        except NotImplementedError:
            flops = calculate_minimum_data_streamed_forward(e3nn_tp=implementation.e3nn_tp, batch_size=self.bench_batch_size)
        

        result += self.calculate_performance_statistics(
            implementation=implementation,
            
            )

        return result
    
    def benchmark_backward(self, implementation : TensorProduct) -> dict:
        result = {
            "tp_direction": "backward",
        }

        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_backward_supplies()

        logger.info("Initialized input / output data.")
        
        time_millis = implementation.benchmark_internal_backward(
            num_warmup=self.num_warmup,
            num_iter=self.num_iter, 
            L1_in=in1, 
            L2_in=in2, 
            weights=weights, 
            L3_grad=out_grad,
            L1_grad=in1_grad, 
            weights_grad=weights_grad,
        )

        pass

    def run(self, test_list : list[tuple]):
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
        output_folder.mkdir(parents=True)

        self.validate_inputs(test_list)
        metadata = self.generate_metadata(test_list)

        with open(os.path.join(output_folder,'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2) 

        for config, implementation, direction, do_correctness, do_benchmark in test_list:
            result = {}
            tc_name = f"{convenience_namer(L1, L2, L3)}, {implementation.name()}, {direction}"
            logger.info(f'Starting {tc_name}.')

            tp = implementation(config)
            if do_correctness:
                if direction == "forward": 
                    correctness_result = self.test_correctness_forward(tp)
                if direction == "backward":
                    correctness_result = self.test_correctness_backward(tp)

            if do_benchmark:
                if direction == "forward":
                    re = self.benchmark_forward(tp)
                if direction == "backward":
                    benchmark_result = self.benchmark_backward(tp)

            L1, L2, L3 = tp.L1, tp.L2, tp.L3
            rnames= [rep.to_string().replace(' ', '') for rep in [L1, L2, L3]]
            result += {
                "config": rnames,
                "direction": direction, 
                "name": implementation.name(),
                "correctness": correctness_result,
                "benchmark": benchmark_result
            }
    
            fname = pathlib.Path(f"{output_folder}/{rnames[0]}_{rnames[1]}_{rnames[2]}_{implementation.name()}.json")

            with open(fname, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f'Finished {tc_name}.')

def debug(tp_impl, config : e3nn.o3.TensorProduct, direction="forward"):
    # THESE ARE NOW E3NN IRREPS
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out 
    batch_size = 1
    tp = tp_impl(config)

    rng = np.random.default_rng(12345)

    if direction == "forward":
        L1_in, L2_in, weights, L3_out = get_random_forward_supplies(config, batch_size, prng_seed=12345)

        tp.exec_tensor_product_cpu(L1_in, L2_in, weights, L3_out)

        ground_truth_out = tp.e3nn_tp(
            torch.Tensor(L1_in),
            torch.Tensor(L2_in),
            torch.Tensor(weights)
            ).numpy(force=True)

        print(L3_out - ground_truth_out)

    elif direction == "backward":
        L1_in, L2_in, L3_grad, weights, weights_grad, L1_in_grad, L2_in_grad = get_random_backward_supplies(config, batch_size, prng_seed=12345)


        tp.backward_cpu(L1_in, L1_in_grad, L2_in, L2_in_grad, L3_grad, weights, weights_grad)

        torch_L1_in = torch.Tensor(L1_in, requires_grad=True)
        torch_L2_in = torch.Tensor(L2_in, requires_grad=True)
        torch_weights = torch.Tennsor(weights, requires_grad=True)

        torch_out = tp.e3nn_tp(torch_L1_in, torch_L2_in, torch_weights)

        torch_out.backward(L3_grad)

        ground_truth_L1_in_grad = torch_L1_in.grad.numpy(force=True)
        ground_truth_L2_in_grad = torch_L2_in.grad.numpy(force=True)
        ground_truth_out_weights_grad = torch_weights.grad.numpy(force=True)

        print(L1_in_grad - ground_truth_L1_in_grad)
        print(L2_in_grad - ground_truth_L2_in_grad)
        print(weights_grad - ground_truth_out_weights_grad)
    else:
        assert(False)

if __name__=='__main__':
    FCTP = e3nn.o3.FullyConectedTensorProduct
    default_tests = [ FCTP(in1, in2, out) for in1, in2, out in        
        [
        ("1x5e", "1x5e", "1x3e"),
        ("1x2e", "1x2e", "1x2e"),
        ("1x4e", "1x3e", "1x1e"),
        ("1x4e", "1x3e", "1x5e"),
        ]
    ]

    multiplicity_tests = [ FCTP(in1, in2, out) for in1, in2, out in 
        [
        ("1x4e", "1x3e", "1x3e")
        ("2x4e", "1x3e", "2x5e"),
        ("1x4e", "2x3e", "2x5e"),
        ("2x4e", "2x3e", "4x5e"),
        ]
    ]

    limited_decomp_tests = [ FCTP(in1, in2, out) for in1, in2, out in
        [
        ("32x5e", "1x5e", "32x3e"),
        ("32x3e + 32x2e", "1x0e + 1x1e", (32 * Irreps.spherical_harmonics(3, p=1)).sort().irreps.simplify()),
        ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", (32 * Irreps.spherical_harmonics(3, p=1))).sort().irreps.simplify(), 
        ("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", (32 * Irreps.spherical_harmonics(3, p=1)).sort().irreps.simplify)
        ]
    ]

    bench_suite = TestBenchmarkSuite(limited_decomp_tests, bench_batch_size=1000000)
    bench_suite.run([MultiplicityOuterProductTP])

    #debug(LoopUnrollTP, ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3))
    #debug(LoopUnrollTP, ("32x5e", "1x5e", "32x3e"), direction="backward")
