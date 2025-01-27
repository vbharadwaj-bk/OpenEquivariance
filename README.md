# OpenEquivariance

[[Examples]](#show-me-some-examples) [[Installation]](#installation)
[[Supported Tensor Products]](#tensor-products-we-accelerate)
[[Citation and Acknowledgements]](#citation-and-acknowledgements)

OpeqnEquivariance is a kernel generator for the Clebsch-Gordon tensor product, 
a key kernel in rotation-equivariant deep neural networks. 
It implements some of the tensor products 
that [e3nn](https://e3nn.org/) supports that are
commonly found in graph neural networks 
(e.g. [Nequip](https://github.com/mir-group/nequip) or
[MACE](https://github.com/ACEsuit/mace)). 

We provide up to an order of magnitude acceleration over e3nn
and up to ~2x speedup over 
[NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. We also offer fused
equivariant graph convolutions that can reduce 
computation memory consumption significantly. 

We currently support NVIDIA GPUs and offer a torch frontend.
HIP support for AMD is planned!

**Warning**: This is an early release, bug reports are welcome.

## Show me some examples
Here's a CG tensor product implemented by e3nn: 

```python
import torch
import e3nn.o3 as o3

gen = torch.Generator(device='cuda')

batch_size = 1000
X_ir, Y_ir, Z_ir = o3.Irreps("1x2e"), o3.Irreps("1x3e"), o3.Irreps("1x2e") 
X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)

instructions=[(0, 0, 0, "uvu", True)]

tp_e3nn = o3.TensorProduct(X_ir, Y_ir, Z_ir, instructions,
        shared_weights=False, internal_weights=False).to('cuda')
W = torch.rand(batch_size, tp_e3nn.weight_numel, device='cuda', generator=gen)

Z = tp_e3nn(X, Y, W)
print(torch.norm(Z))
```

And here's the same tensor product using openequivariance. We require that your
tensors are stored on a CUDA device for this to work: 

```python
import openequivariance as oeq

problem = oeq.TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
tp_fast = oeq.LoopUnrollTP(problem, torch_op=True)

Z = tp_fast(X, Y, W) # Reuse X, Y, W from earlier
print(torch.norm(Z))
```

Our interface for `oeq.TPProblem` is almost a strict superset of 
`o3.TensorProduct` (two key differences: we 
impose `internal_weights=False` and add support for multiple datatypes). 
You can pass e3nn `Irreps` instances directly or 
use `oeq.Irreps`, which is identical. 

We recommend reading the [e3nn documentation and API reference](https://docs.e3nn.org/en/latest/) first, then using our kernels 
as drop-in replacements. We support most "uvu" and "uvw" tensor products; 
see [this section](#tensor-products-we-accelerate) for an up-to-date list of supported configurations. 

**Important**: For many configurations, our kernels return results identical to
e3nn up to floating point roundoff (this includes all "uvu" problems with
multiplicity 1 for all irreps in the second input). For other configurations 
(e.g. any "uvw" connection modes), we return identical 
results up to a well-defined reordering of the weights relative to e3nn. 

If you're executing tensor products as part of a message passing graph
neural network, we offer fused kernels that save both memory and compute time (only supported
for "uvu" at the moment, "uvw" support coming soon): 

```python
from torch_geometric import EdgeIndex

node_ct, nonzero_ct = 3, 4

# Receiver, sender indices for message passing GNN
edge_index = EdgeIndex(
                [[0, 1, 1, 2],  # Receiver 
                 [1, 0, 2, 1]], # Sender 
                device='cuda',
                dtype=torch.long)

X = torch.rand(node_ct, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(nonzero_ct, Y_ir.dim, device='cuda', generator=gen)
W = torch.rand(nonzero_ct, problem.weight_numel, device='cuda', generator=gen)

tp_conv = oeq.LoopUnrollConv(problem, torch_op=True, deterministic=False) # Reuse problem from earlier
Z = tp_conv.forward(X, Y, W, edge_index[0], edge_index[1]) # Z has shape [node_ct, z_ir.dim]
print(torch.norm(Z))
```

If you can guarantee `EdgeIndex` is sorted by receiver index and supply the transpose
permutation, we can provide even greater speedup (and deterministic results) 
by avoiding atomics: 

```python
_, sender_perm = edge_index.sort_by("col")            # Compute transpose perm 
edge_index, receiver_perm = edge_index.sort_by("row") # Sort by receiver index

# Now we can use the faster deterministic algorithm
tp_conv = oeq.LoopUnrollConv(problem, torch_op=True, deterministic=True) 
Z = tp_conv.forward(X, Y[receiver_perm], W[receiver_perm], edge_index[0], edge_index[1], sender_perm) 
print(torch.norm(Z))
```
**Note**: you don't need Pytorch geometric to use our kernels. When
`deterministic=False`, the `sender` and `receiver` indices can have
arbitrary order. 

## Installation 
Right now, we only support source builds, 
but we provide scripts to streamline installation.

We highly recommend that you use
`conda` or `mamba` to set up a Python environment for installation.

### Build via install script and pip (fastest) 
The steps below assume that you're using a bash shell and have a C / C++ 
compiler that CMake can find. If not, you can install [gxx](https://anaconda.org/conda-forge/gxx/) from `conda-forge`. 

1. **Setup**: Create an environment (or activate an existing one) with 
  our core dependencies: 
    ```bash
    conda create -c conda-forge --name my_env python=3.11 pybind11 cmake nvidia::cuda-toolkit
    conda activate my_env 
    ``` 

2. **Install**: Build our package and install via `pip`: 
    ```bash
    git clone https://github.com/vbharadwaj-bk/OpenEquivariance
    cd OpenEquivariance 
    sh dev_build.sh 
    pip install . # Use pip install -e . for an editable install 
    ``` 

3. **Test**: You're ready to go!

You don't have to install NVIDIA's CUDA toolkit or CMake if they exist on your
platform, but you're responsible for setting LD_LIBRARY_PATH so that libraries
are findable at runtime. Installing the CUDA toolkit via `conda` takes care of this for
you. 

### Build to replicate our benchmarks 
To run our benchmark suite, you'll also need the following packages: 
- `e3nn`, 
- `cuEquivariance`
- `cuEquivariance-torch` 
- `cuEquivariance-ops-torch-cu11` OR `cuEquivariance-ops-torch-cu12` 
- `matplotlib` (to reproduce our figures) 

We conducted our benchmarks on an NVIDIA A100-SXM-80GB GPU at
Lawrence Berkeley National Laboratory. Your results may differ 
a different GPU.

### conda or mambabuild (experimental)
We are experimenting with setup using `conda-build` or
`mambabuild`. Stay tuned for that

## Tensor products we accelerate 
e3nn supports a variety of connection modes for CG tensor products. We support 
two that are commonly used in equivariant graph neural networks:
"uvu" and "uvw". Our JIT compiled kernels should handle:

1. Pure "uvu" tensor products, which are most efficient when the input with higher
multiplicities is the first argument. Our results are identical to e3nn when irreps in
the second input have multiplicity 1, and otherwise identical up to a reordering
of the input weights.

2. Pure "uvw" tensor products, which are currently more efficient when the input with
higher multiplicities is the first argument. Our results are identical to e3nn up to a reordering
of the input weights. 

Our code includes correctness checks, but the configuration space is large. If you notice
a bug, let us know in a Github issue. We'll try our best to correct it or document the problem here.

We do not (yet) support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".
- Non-trainable instructions: all of your instructions must have weights associated. 

If you have a use case for any of the unsupported features above, let us know.

## Citation and Acknowledgements
If you find this code useful, please cite us.

[Citation coming soon!]

Our codebase includes a lightweight clone of 
[e3nn](https://e3nn.org/)'s frontend interface (in particular, the 
`TensorProduct` and `Irreps` classes). We removed references to Pytorch
and separated the implementation from the problem description (for future
frontend support outside of torch). Thank you to the current
developers and maintainers! 
