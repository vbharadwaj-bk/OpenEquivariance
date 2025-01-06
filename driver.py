import itertools, typing, random

import numpy as np
import numpy.linalg as la

from src.benchmark.logging_utils import *
from src.implementations.e3nn_lite import *
from src.benchmark.e3nn_lite_utils import *
from build.kernel_wrapper import *
from src.benchmark.random_buffer_utils import get_random_buffers_forward, get_random_buffers_backward
from src.benchmark.TestBenchmarkSuite import TestBenchmarkSuite, TestDefinition, Direction
from src.benchmark.tpp_creation_utils import *
from src.implementations.LoopUnrollTP import LoopUnrollTP
from src.implementations.NumpyTensorProduct import NumpyTensorProduct
from src.implementations.MultiplicityOuterProductTP import MultiplicityOuterProductTP
from src.implementations.LoopReorderUVWTP import LoopReorderUVWTP
from src.implementations.ManyOneUVWTP import ManyOneUVWTP 
from src.implementations.E3NNTensorProduct import E3NNTensorProduct
from src.implementations.CUETensorProduct import CUETensorProduct

logger = getLogger()

def debug( test : TestDefinition) -> None: 
    print(test)
    tp_impl = test.implementation
    config = test.problem  
    direction = test.direction

    assert issubclass(tp_impl, TensorProduct)
    assert isinstance(config, TPProblem)
    assert direction in typing.get_args(Direction)

    batch_size = 1
    # prng_seed = 12345
    prng_seed = random.randint(100, 100000)

    tp = tp_impl(config)

    from src.implementations.E3NNTensorProduct import E3NNTensorProduct
    ref_tp = E3NNTensorProduct(config)

    logger.debug(repr(config))

    np.set_printoptions(
        formatter={'float': '{:0.3f}'.format},
        linewidth=200,
        threshold=sys.maxsize,
        )
    
    if direction == "forward":
        in1, in2, weights, out = get_random_buffers_forward(tpp=config, batch_size=batch_size, prng_seed=prng_seed)

        out_shape = (-1, config.irreps_out[0].ir.dim)
        # print("out")
        # print(out)
        test_out = out.copy()
        tp.forward_cpu(
            L1_in=in1, 
            L2_in=in2, 
            L3_out=test_out, 
            weights=weights
            ) 
          
        print("test out")
        print(test_out.reshape(out_shape))

        ground_truth_out = out.copy()
        ref_tp.forward_cpu(
            L1_in=in1, 
            L2_in=in2, 
            L3_out=ground_truth_out,
            weights=weights
            )

        print("ground truth out")
        print(ground_truth_out.reshape(out_shape))

        print("LA.Norm:")
        print(la.norm((test_out - ground_truth_out).flatten(), ord=np.inf))

        print("test_output / ground_truth_output")
        print( (test_out / ground_truth_out).reshape(out_shape))

    elif direction == "backward":
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward(tpp=config, batch_size=batch_size, prng_seed=prng_seed)

        test_in1_grad = in1_grad.copy()
        test_in2_grad = in2_grad.copy()
        test_weights_grad = weights_grad.copy()
        tp.backward_cpu(
            L1_in=in1.copy(),
            L1_grad=test_in1_grad,
            L2_in=in2.copy(),
            L2_grad=test_in2_grad,
            L3_grad=out_grad.copy(),
            weights=weights.copy(),
            weights_grad=test_weights_grad            
        )

        ref_in1_grad = in1_grad.copy()
        ref_in2_grad = in2_grad.copy()
        ref_weights_grad = weights_grad.copy()

        ref_tp.backward_cpu(
            L1_in=in1.copy(),
            L1_grad=ref_in1_grad, 
            L2_in=in2.copy(), 
            L2_grad=ref_in2_grad,
            L3_grad=out_grad.copy(), 
            weights=weights.copy(),
            weights_grad=ref_weights_grad, 
        )


        for name, ground_truth, test_result in [
            ("weight_grad", ref_weights_grad, test_weights_grad),
            ("L1_grad", ref_in1_grad , test_in1_grad),
            ("L2_grad", ref_in2_grad , test_in2_grad),
            ]:
            print(name)
            print("ground truth")
            print(ground_truth)
            print("test results")
            print(test_result)
            print("ratio")
            print(test_result / ground_truth)
            print("LA.Norm:")
            print(la.norm((test_result - ground_truth).flatten(), ord=np.inf))
            print()
    else:
        assert(False)
    np.set_printoptions()

if __name__=='__main__':  
    FCTPP = FullyConnectedTPProblem
    ChannelTPP = ChannelwiseTPP 
    basic_fully_connected_problems = [
        FCTPP("1x1e", "1x1e", "1x1e"),
        FCTPP("1x1e", "1x1e", "4x1e"),
        FCTPP("1x1e", "2x1e", "1x1e"), 
        FCTPP("2x1e", "1x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "2x1e"),
        FCTPP("2x1e", "2x1e", "4x1e") 
    ]

    increasing_multiplicty_fully_connected_problems = [
        FCTPP("2x1e", "2x1e", "2x1e"),
        FCTPP("4x1e", "4x1e", "4x1e"),
        FCTPP("8x1e", "8x1e", "8x1e"),
        FCTPP("16x1e", "16x1e", "16x1e"),
        FCTPP("32x1e", "32x1e", "32x1e"),
    ]

    cutlass_troubleshooting = [
        FCTPP("16x1e", "16x1e", "16x1e"),
        FCTPP("15x1e", "16x1e", "16x1e"), 
        FCTPP("16x1e", "15x1e", "16x1e"), 
        FCTPP("16x1e", "17x1e", "16x1e"),
        FCTPP("17x1e", "16x1e", "16x1e"),
        FCTPP("16x1e", "16x1e", "15x1e"),
        FCTPP("16x1e", "16x1e", "17x1e"),
        FCTPP("16x1e", "16x1e", "31x1e"),
        FCTPP("16x1e", "16x1e", "32x1e"),
        FCTPP(" 1x1e", "16x1e", "32x1e"),
        FCTPP(" 1x1e", " 1x1e", "32x1e"),
        FCTPP(" 1x1e", " 1x1e", "31x1e"),
        FCTPP(" 1x1e", " 1x1e", "16x1e"),
        FCTPP(" 1x1e", " 1x1e", "15x1e"),
        FCTPP(" 1x1e", " 1x1e", "14x1e"),
        FCTPP(" 1x1e", " 1x1e", "13x1e"),
        FCTPP(" 1x1e", " 1x1e", "12x1e"),

        FCTPP(" 1x1e", " 1x1e", " 9x1e"),
        FCTPP(" 1x1e", " 1x1e", " 8x1e"),
        FCTPP(" 1x1e", " 1x1e", " 4x1e"),
        FCTPP(" 1x1e", " 1x1e", " 2x1e"),
        FCTPP(" 1x1e", " 1x1e", " 1x1e"),
        
        FCTPP(" 1x1e", " 1x1e", "11x1e"),
        FCTPP(" 1x1e", " 1x1e", "10x1e"),
        
        FCTPP(" 1x1e", " 1x1e", " 7x2e"),
        FCTPP(" 1x1e", " 1x1e", " 6x2e"),
        
        FCTPP(" 1x1e", " 1x2e", " 9x3e"),
        FCTPP(" 1x1e", " 1x2e", " 8x3e"),
    ]

    full_size_uvw_case = [
        FCTPP("32x1e", "32x1e", "32x1e"),
        FCTPP("32x2e", "32x2e", "32x2e"),
        FCTPP("32x3e", "32x3e", "32x3e"),
        FCTPP("32x4e", "32x4e", "32x4e"),
        FCTPP("32x5e", "32x5e", "32x5e"),
    ]

    basic_multi_interaction_problems = [
        FCTPP("2x1e + 1x0e", "2x1e", "4x1e"),
        FCTPP("2x1e", "2x1e + 1x0e", "4x1e"),
        FCTPP("2x1e + 1x0e", "2x1e + 1x0e", "4x1e"),
        FCTPP("32x1e + 32x0e", "32x1e + 32x0e", "32x1e + 32x0e"),
    ]

    conv_problems = [  
        #FCTPP("32x2e", "32x1e", "32x2e"),
        #SingleInstruction("32x5e", "1x3e", "32x5e", "uvu", True),
        #SingleInstruction("32x5e", "1x5e", "32x3e", "uvu", True)
        #mace_conf("64x2e", "1x0e", 2), 
        #mace_conf("128x1o + 128x0e", "1x0e + 1x1e + 1x2e + 1x3e", 2),
        #mace_conf("128x0e", "1x0e + 1x1e + 1x2e + 1x3e", 2), 
        ChannelTPP("128x2e + 128x1e + 128x0e", "1x0e + 1x1e + 1x2e + 1x3e", 2)
    ]

    #for problem in conv_problems:
    #    problem.irrep_dtype = np.float64
    #    problem.weight_dtype = np.float64

    problems = list(itertools.chain(
        basic_fully_connected_problems,
        increasing_multiplicty_fully_connected_problems,
        full_size_uvw_case,
        cutlass_troubleshooting,
        basic_multi_interaction_problems,
        #conv_problems,
    ))
 
    implementations = [
        #E3NNTensorProduct,
        #CUETensorProduct, 
        #LoopUnrollTP,
        LoopReorderUVWTP,
        # MultiplicityOuterProductTP,
        ]
    
    directions : list[Direction] = [
        'forward', 
        # 'backward',
        ] 

    tests = [TestDefinition(implementation, problem, direction, correctness=True, benchmark=False) 
             for problem, direction, implementation
             in itertools.product(problems, directions, implementations)]
 
    bench_suite = TestBenchmarkSuite(
        correctness_threshold = 5e-5,
        num_iter=5,
        bench_batch_size=10_000_000,
        #reference_implementation=NumpyTensorProduct,
        prng_seed=11111
    )

    logger.setLevel(logging.DEBUG)
    bench_suite.run(tests)
    # debug(tests[4])