import e3nn
from e3nn import o3

import sys, os
from build.kernel_wrapper import *
from src.implementations.LoopUnrollTP import *

def compare_output_to_e3nn(config, batch_size):
    reps = None
    if isinstance(config[2], int): 
        reps = RepTriple(Representation(config[0]), Representation(config[1]), config[2])
    else:
        reps = RepTriple(Representation(config[0]), Representation(config[1]), Representation(config[2]))

    instructions = [reps.interactions(i) for i in range(reps.num_interactions())]
    instructions = [(i[0], i[1], i[2], "uvu", False) for i in instructions]

    irreps_L1 = o3.Irreps(reps.L1.to_string())
    irreps_L2 = o3.Irreps(reps.L2.to_string())
    irreps_L3 = o3.Irreps(reps.L3.to_string()) 

    fast_tp = LoopUnrollTP(reps, batch_size)
    tp = o3.TensorProduct(irreps_L1, irreps_L2, irreps_L3, instructions=instructions) # internal_weights=False, shared_weights=False
    print(tp.weight_numel)

    In1 = irreps_L1.randn(batch_size, -1)
    In2 = irreps_L2.randn(batch_size, -1)

    In1.requires_grad_()
    In2.requires_grad_()

    L3_out_fast = np.zeros((1, reps.L3.get_rep_length() ), dtype=np.float32)

    fast_tp.exec_tensor_product_cpu(In1.detach().numpy(), In2.detach().numpy(), L3_out_fast)
    e3nn_output = tp(In1, In2)

    print("Forward Comparison")
    #print(e3nn_output.detach().numpy() / L3_out_fast) 

    # ======= Backward ========
    random_grad = irreps_L3.randn(1, -1)
    e3nn_output.backward(gradient=random_grad, inputs=[In1, In2])
    weights = np.ones((1, reps.num_trainable_weights()), dtype=np.float32) # We set the weights to all ones for now 
    L1_grad, L2_grad, weights_grad = fast_tp.backward_cpu(In1.detach().numpy(), In2.detach().numpy(), random_grad.detach().numpy(), weights)

    print("Backward Comparison")
    print(In1.grad.numpy() / L1_grad)
    print(In2.grad.numpy() / L2_grad)
