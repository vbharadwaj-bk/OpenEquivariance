import torch
import numpy as np
from build.kernel_wrapper import *

class IrrepTransposer:
    next_id = 0

    def __init__(self, rep):
        self.rep = Representation(str(rep))
        self.id = IrrepTransposer.next_id
        IrrepTransposer.next_id += 1

        @torch.library.custom_op(f"fast_tp::transpose_forward{self.id}", mutates_args=(), device_types="cuda")
        def forward(input : torch.tensor, row_major: bool) -> torch.tensor:
            irreps_to_transpose = np.array(input.contiguous().detach().cpu().numpy(), copy=True)
            self.rep.transpose_irreps_cpu(irreps_to_transpose, row_major)
            return torch.tensor(irreps_to_transpose, device='cuda')

        @forward.register_fake
        def _(input, row_major):
            return input.empty_like(input.shape)

        self.forward = forward

        def setup_context(ctx, inputs, output):
            ctx.row_major = inputs[1] 
 
        self.forward.register_autograd(lambda ctx, grad_output: (forward(grad_output, not ctx.row_major), None), 
                setup_context=setup_context)

    def __call__(self, input, row_major):
        return self.forward(input, row_major)