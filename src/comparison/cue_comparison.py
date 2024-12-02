import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

e = cue.descriptors.channelwise_tensor_product(
    cue.Irreps("O3", "128x0e + 128x1o + 128x2e"),
    cue.Irreps("O3", "1x0e + 1x1e"))
    #cue.Irreps("O3", "32x5e"),
    #cue.Irreps("O3", "1x3e"),
    #cue.Irreps("O3", "32x5e"))

module = cuet.EquivariantTensorProduct(e, layout=cue.ir_mul)
module.to('cuda')

print(module)

batch=100000

w = torch.randn((batch, e.inputs[0].irreps.dim)).to('cuda')
x = torch.randn((batch, e.inputs[1].irreps.dim)).to('cuda')
y = torch.randn((batch, e.inputs[2].irreps.dim)).to('cuda')

#w.requires_grad_()
#x.requires_grad_()
#y.requires_grad_()
results = []

for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    results.append(module(w, x, y, use_fallback=False))
    end.record()
    torch.cuda.synchronize() 
    print(start.elapsed_time(end))

#grad = torch.randn(*result.shape).to('cuda')

#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)

#start.record()
#result.backward(gradient=grad)
#end.record()

#torch.cuda.synchronize()
#print(start.elapsed_time(end))