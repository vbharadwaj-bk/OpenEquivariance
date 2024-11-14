import numpy as np

from src.benchmark.supplies_utils import get_random_supplies_forward, get_random_supplies_backward
from src.benchmark.perf_metrics_utils import (
    calculate_minimum_flops_forward, 
    calculate_minimum_memory_streamed_forward, 
    calculate_minimum_memory_streamed_backward,
    )
from src.benchmark.e3nn_lite_utils import calculate_total_nnz
from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import TPProblem

from src.benchmark.logging_utils import getLogger, bcolors 

logger = getLogger()

def calculate_performance_statistics( 
            problem : TPProblem, 
            batch_size : int, 
            total_flops : int, 
            total_memory_streamed : int, 
            time_millis : np.ndarray,
            ) -> dict:
        result = {}

        throughputs_gflops = [float(x) for x in total_flops / (time_millis * 1e6)]
       
        bandwidth_gbps = [float(x) for x in total_memory_streamed / (time_millis * 1e6)]

        nnz = calculate_total_nnz(problem)

        time_millis = [float(x) for x in time_millis]

        result |= {
            "total_cg_nnz": nnz,
            "flops_per_tp": str(total_flops / batch_size),
            "L1": str(problem.irreps_in1),
            "L2": str(problem.irreps_in2),
            "L3": str(problem.irreps_out),

            "L1_rep_len": problem.irreps_in1.dim,
            "L2_rep_len": problem.irreps_in2.dim,
            "L3_rep_len": problem.irreps_out.dim,

            "rep_dtype": "float", # If this changes, also need to modify the arithmetic intensity calculation
            "arithmetic_intensity (FLOPs / byte)": str(total_flops * total_memory_streamed), 

            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps": bandwidth_gbps,
        }

        logger.info(f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")

        return result 

def benchmark_forward(
        problem : TPProblem, 
        implementation : type[TensorProduct],
        batch_size : int, 
        num_warmup : int,
        num_iter : int, 
        prng_seed : int, 
        ) -> dict:
    '''
    This function sets up the necessary materials and calls the internal benchmarker
    '''
    result = {
        "tp_direction": "forward",
        "num_warmup": num_warmup,
        "num_iter": num_iter,
        "prng_seed": prng_seed,
    }

    L1_in, L2_in, weights, L3_buffer = get_random_supplies_forward(problem, batch_size, prng_seed)

    logger.info("Initialized input / output data.")

    tp = implementation(problem)

    # BENCHMARK 
    try: 
        time_millis = tp.benchmark_forward(
            num_warmup=num_warmup, 
            num_iter=num_iter,
            L1_in=L1_in,
            L2_in=L2_in,
            weights=weights,
            L3_buffer=L3_buffer
            )
    except NotImplementedError:
        logger.warning("Benchmarking is not implemented, time millis replaced with -1's")
        time_millis = np.full(shape=num_iter, fill_value=-1) 
    
    # FLOPS 
    try:
        flops = tp.calculate_flops_forward(batch_size=batch_size)
    except NotImplementedError:
        logger.warning("Actual flops was not calcuated, so minimum values are being used")
        flops = calculate_minimum_flops_forward(problem, batch_size=batch_size)
    
    # DATA
    try: 
        memory_streamed = tp.calculate_memory_streamed_backward(batch_size=batch_size)
    except NotImplementedError: 
        logger.warning("Actual memory streamed was not calcuated, so minimum values are being used")
        memory_streamed = calculate_minimum_memory_streamed_forward(problem, batch_size=batch_size)
             

    result |= calculate_performance_statistics(
        problem=problem,
        batch_size=batch_size,
        total_flops=flops["total"],
        total_memory_streamed=memory_streamed["total"],
        time_millis=time_millis
        )

    return result       

def benchmark_backward(
        problem : TPProblem, 
        implementation : type[TensorProduct],
        batch_size : int, 
        num_warmup : int,
        num_iter : int, 
        prng_seed : int, 
        ) -> dict:
        
        result = {
            "tp_direction": "backward",
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
        }

        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_supplies_backward(problem, batch_size, prng_seed)
    
        logger.info("Initialized input / output data.")
        
        try:
            time_millis = implementation.benchmark_internal_backward(
                num_warmup=num_warmup,
                num_iter=num_iter, 
                L1_in=in1, 
                L2_in=in2, 
                weights=weights, 
                L3_grad=out_grad,
                L1_grad=in1_grad, 
                weights_grad=weights_grad,
            )
        except NotImplementedError:
            logger.warning("Benchmarking is not implemented, time millis replaced with -1's")
            time_millis = np.full(shape=num_iter, fill_value=-1)
         
        ### GETTING FLOPS AND MEMORY STREAMED WITH FALLBACKS FOR NOT IMPLEMENTED CALCUATIONS

        try:
            flops = implementation.calculate_flops_forward(batch_size=batch_size)
        except NotImplementedError:
            try:
                flops = calculate_minimum_flops_forward(e3nn_tp=implementation.e3nn_tp, batch_size=batch_size)
                logger.warning("Actual flops was not calcuated, so minimum values are being used")
            except NotImplementedError: 
                logger.warning("Minimum Backwards flops calcuations are not implemented, -1 is a placholder")
                flops = {"total" : -1}
        
        try: 
            memory_streamed = implementation.calculate_memory_streamed_backward(batch_size=batch_size)
        except NotImplementedError: 
            logger.warning("Actual memory streamed was not calcuated, so minimum values are being")
            memory_streamed = calculate_minimum_memory_streamed_backward(tpp=problem, batch_size=batch_size)
        
        result |= calculate_performance_statistics(
            problem=problem,
            batch_size=batch_size,
            total_flops=flops["total"],
            total_memory_streamed=memory_streamed["total"],
            time_millis=time_millis
            )

        return result