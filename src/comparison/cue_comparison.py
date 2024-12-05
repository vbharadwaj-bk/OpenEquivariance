import torch
import cuequivariance as cue
import cuequivariance_torch as cuet
import time


e = cue.descriptors.channelwise_tensor_product(
    cue.Irreps("SO3", "128x0+128x1+128x2"),
    cue.Irreps("SO3", "1x0+1x1+1x2+1x3"),
    cue.Irreps("SO3", "128x0+128x1+128x2"),
)

module = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul)
module.to('cuda')

print(module)

batch=10000
w = torch.randn((batch, e.inputs[0].irreps.dim)).to('cuda')
x = torch.randn((batch, e.inputs[1].irreps.dim)).to('cuda')
y = torch.randn((batch, e.inputs[2].irreps.dim)).to('cuda')

w.requires_grad_()
x.requires_grad_()
y.requires_grad_()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in range(10):
    result = module(w, x, y, use_fallback=False)
    grad = torch.randn(*result.shape, device='cuda')
    result = module(w, x, y, use_fallback=False)
    start.record()
    result.backward(gradient=grad)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))