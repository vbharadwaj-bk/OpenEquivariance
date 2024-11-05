import pathlib, os, json, time

import e3nn 
import torch

from src.implementations.TensorProduct import TensorProduct
from src.benchmark.logging_utils import getLogger, bcolors 
from src.benchmark.e3nn_tp_utils import *

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
        assert isinstance(test_list, list)
        assert len(test_list) != 0
        for config, implementation, direction, do_correctness, do_benchmark in test_list:
            assert isinstance(config, e3nn.o3.TensorProduct)
            assert issubclass(implementation, TensorProduct)
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
            config_names.add(convenience_namer(config.irreps_in1, config.irreps_in2, config.irreps_out)) 
            implementation_names.add(implementation.name())
            directions.add(direction)
            did_correctness = did_correctness or do_correctness
            did_benchmark = did_benchmark or do_benchmark 

        metadata = {
                "configs": list(config_names),
                "implementations": list(implementation_names),
                "directions": list(directions),
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
        
        ### GETTING FLOPS AND MEMORY STREAMED WITH FALLBACKS FOR NOT IMPLEMENTED CALCUATIONS

        try:
            flops = implementation.calculate_flops_forward(batch_size=self.bench_batch_size)
        except NotImplementedError:
            flops = calculate_minimum_flops_forward(e3nn_tp=implementation.e3nn_tp, batch_size=self.bench_batch_size)
        
        try: 
            memory_streamed = implementation.calculate_memory_streamed_backward(batch_size=self.bench_batch_size)
        except NotImplementedError: 
            memory_streamed = calculate_minimum_memory_streamed_forward(e3nn_tp=implementation.e3nn_tp, batch_size=self.bench_batch_size)

        result += self.calculate_performance_statistics(
            implementation=implementation,
            total_flops=flops["total"],
            total_memory_streamed=memory_streamed["total"],
            time_millis=time_millis
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
         
        ### GETTING FLOPS AND MEMORY STREAMED WITH FALLBACKS FOR NOT IMPLEMENTED CALCUATIONS

        try:
            flops = implementation.calculate_flops_forward(batch_size=self.bench_batch_size)
        except NotImplementedError:
            flops = calculate_minimum_flops_forward(e3nn_tp=implementation.e3nn_tp, batch_size=self.bench_batch_size)
        
        try: 
            memory_streamed = implementation.calculate_memory_streamed_backward(batch_size=self.bench_batch_size)
        except NotImplementedError: 
            memory_streamed = calculate_minimum_memory_streamed_forward(e3nn_tp=implementation.e3nn_tp, batch_size=self.bench_batch_size)
        
        result += self.calculate_performance_statistics(
            implementation=implementation,
            total_flops=flops["total"],
            total_memory_streamed=memory_streamed["total"],
            time_millis=time_millis
            )

        return result

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
            tc_name = f"{convenience_namer(config.irreps_in1, config.irreps_in1, config.irreps_out)}, {implementation.name()}, {direction}"
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