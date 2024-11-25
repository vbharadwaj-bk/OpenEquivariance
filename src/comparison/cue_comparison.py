import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

e = cue.descriptors.channelwise_tensor_product(
    #cue.Irreps("O3", "32x0e + 32x1o + 32x2e"),
    #cue.Irreps("O3", "1x0e + 1x1e"),
    cue.Irreps("O3", "32x5e"),
    cue.Irreps("O3", "1x3e"),
    cue.Irreps("O3", "32x5e"))

module = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul)
module.to('cuda')

print(module)

batch=10000
w = torch.randn((batch, e.inputs[0].irreps.dim)).to('cuda')
x = torch.randn((batch, e.inputs[1].irreps.dim)).to('cuda')
y = torch.randn((batch, e.inputs[2].irreps.dim)).to('cuda')

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
#result = module(w, x, y, use_fallback=False)
end.record()
torch.cuda.synchronize()
 
print(start.elapsed_time(end))