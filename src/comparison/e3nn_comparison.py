import torch, e3nn
from e3nn import o3
import src.implementations.e3nn_lite 

import sys, os
from build.kernel_wrapper import *
from src.implementations.LoopUnrollTP import *
from src.torch_modules.irrep_transposer import *

def test_drive_torch_module():
    torch.set_default_device('cuda')
    irreps_L1 = o3.Irreps("32x5e")
    irreps_L2 = o3.Irreps("1x5e")
    irreps_L3 = o3.Irreps("32x3e")
        
    In1 = irreps_L1.randn(1, -1)
    In2 = irreps_L2.randn(1, -1)
    weights = torch.ones(32, dtype=torch.float32)

    In1.to(device='cuda')
    In2.to(device='cuda')
    weights.to(device='cuda')

    In1.requires_grad_()
    In2.requires_grad_()
    weights.requires_grad_()

    L1T = IrrepTransposer(irreps_L1)
    L3T = IrrepTransposer(irreps_L3)

    fast_tp = LoopUnrollTP(o3.TensorProduct(irreps_L1, irreps_L2, irreps_L3, 
        instructions=[(0, 0, 0, "uvu", True)]
    ), 
    torch_op=True)
    result_ours = L3T(fast_tp.forward(L1T(In1, True), In2, weights), False)

    random_grad = irreps_L3.randn(1, -1)
    result_ours.backward(random_grad, inputs=[In1, In2])
