from typing import Optional

from src.implementations.TensorProduct import TensorProduct
from src.implementations.e3nn_lite import TPProblem
from src.benchmark.random_buffer_utils import get_random_buffers_forward, get_random_buffers_backward
from src.benchmark.logging_utils import getLogger, bcolors 
import numpy as np 
import numpy.linalg as la

logger = getLogger()

def check_similiarity(name : str,  to_check : np.ndarray,  ground_truth : np.ndarray, correctness_threshold : float):
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
        result["pass"] = bool(diff_Linf_norm < correctness_threshold) 

        if result["pass"]:
            logger.info(f"{bcolors.OKGREEN}{name} correctness check pass. {bcolors.ENDC}")
        else:
            logger.error(f"{bcolors.FAIL}{name} correctness check fail! {diff_Linf_norm=}, {correctness_threshold=} {bcolors.ENDC}")

    return result

def correctness_forward(
        problem : TPProblem,  
        test_implementation : type[TensorProduct], 
        reference_implementation : Optional[type[TensorProduct]], 
        batch_size : int, 
        correctness_threshold : float,
        prng_seed : int,
        ) -> dict:
    
    if reference_implementation is None:
        from src.implementations.E3NNTensorProduct import E3NNTensorProduct
        reference_implementation = E3NNTensorProduct

    result = {
        "thresh": correctness_threshold, 
        "batch_size":batch_size
    }

    # run reference
    in1, in2, weights, out = get_random_buffers_forward(problem, batch_size, prng_seed)
    ref_tp = reference_implementation(problem)
    ref_tp.forward_cpu(in1, in2, out, weights)
    ref_out = out.copy()

    # run test
    in1, in2, weights, out = get_random_buffers_forward(problem, batch_size, prng_seed)
    test_tp = test_implementation(problem)
    test_tp.forward_cpu(in1, in2, out, weights)
    test_out = out.copy()
    
    # check similarity 
    for name, to_check, ground_truth in [
        ("output", ref_out, test_out)
        ]:
        result[name] = check_similiarity(name, to_check, ground_truth, correctness_threshold)
    
    return result

def correctness_backward(
        problem : TPProblem,  
        test_implementation : type[TensorProduct], 
        reference_implementation : Optional[type[TensorProduct]], 
        batch_size : int, 
        correctness_threshold : float,
        prng_seed : int,
        ) -> dict:
    
    if reference_implementation is None:
        from src.implementations.E3NNTensorProduct import E3NNTensorProduct
        reference_implementation = E3NNTensorProduct

    result = {
        "thresh": correctness_threshold, 
    }
    
    # run reference
    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward(
        problem, 
        batch_size, 
        prng_seed
    )
    
    ref_tp = reference_implementation(problem)
    ref_tp.backward_cpu(in1, in1_grad, in2, in2_grad, out_grad, weights, weights_grad) 

    ref_weights_grad = weights_grad.copy()
    ref_in1_grad = in1_grad.copy()
    ref_in2_grad = in2_grad.copy()

    # run test version

    in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward(
        problem, 
        batch_size, 
        prng_seed
    )

    test_tp = test_implementation
    test_tp.backward_cpu(in1, in1_grad,in2, in2_grad, weights, weights_grad)
    
    test_weights_grad = weights.copy()
    test_in1_grad = in1_grad.copy()
    test_in2_grad = in2_grad.copy()
    
    ## CHECK OUTPUT SIMILARITY 
    for name, to_check, ground_truth in [
        ("weight_grad", test_weights_grad, ref_weights_grad),
        ("in1_grad", test_in1_grad, ref_in1_grad),
        ("in2_grad", test_in2_grad, ref_in2_grad),
        ]:
        result[name] = check_similiarity(name, to_check, ground_truth, correctness_threshold)
    
    return result   