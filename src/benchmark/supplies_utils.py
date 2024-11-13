import numpy as np

from src.implementations.e3nn_lite import TPProblem

def get_random_supplies_forward(tpp : TPProblem, batch_size : int,  prng_seed : int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return properly sized numpy arrays needed to execute a tensor product in the forward direction
    Supports shared vs non-shared weights
    """
    assert isinstance(tpp, TPProblem)
    rng = np.random.default_rng(prng_seed)

    in1 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in1.dim)), dtype=np.float32) 
    in2 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in2.dim)), dtype=np.float32)

    num_weights = tpp.weight_numel if tpp.shared_weights else tpp.weight_numel * batch_size
    weights = np.array(rng.uniform(size=(num_weights)), dtype=np.float32)

    out = np.zeros(shape=(batch_size * tpp.irreps_out.dim), dtype=np.float32)

    return in1, in2, weights, out 

def get_random_supplies_backward(tpp : TPProblem, batch_size : int, prng_seed : int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return properly sized numpy arrays needed to execute a tensor product in the backward direction
    Supports shared vs non-shared weights
    """
    assert isinstance(tpp, TPProblem)
    rng = np.random.default_rng(prng_seed)
    
    in1 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in1.dim)), dtype=np.float32) 
    in2 = np.array(rng.uniform(size=(batch_size, tpp.irreps_in2.dim)), dtype=np.float32)
    out_grad = np.array(rng.uniform(size=(batch_size, tpp.irreps_out.dim)), dtype=np.float32)

    num_weights = tpp.weight_numel if tpp.shared_weights else tpp.weight_numel * batch_size
    weights = np.array(rng.uniform(size=(num_weights)), dtype=np.float32)

    weights_grad = np.zeros_like(weights)
    in1_grad = np.zeros_like(in1)
    in2_grad = np.zeros_like(in2) 

    return in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad