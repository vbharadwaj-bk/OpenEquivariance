# ESPMM
This repository a kernel generator for the Clebsch-Gordon tensor product, 
a key kernel in equivariant deep neural networks. It implements
a subset of the functionality of [e3nn](https://e3nn.org/)
for its common use cases in equivariant graph neural networks
(e.g. [Nequip](https://github.com/mir-group/nequip) or
[MACE](https://github.com/ACEsuit/mace)). 

We can provide an order of magnitude acceleration over e3nn
and up to ~2x speedup over 
[NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. We also offer fused
equivariant graph convolutions that can reduce memory consumption 
significantly. 

We currently support NVIDIA GPUs; HIP support for AMD is planned! 

## Show me an example 
Here's a tensor product that appears in MACE, implemented in
e3nn: 

```python
To fill 
```

And here's our code:

```python
To fill 
```

If you're performing a tensor product as part of a graph 
convolution, you can fuse the two operations together to reduce both memory and compute time: 

```python
To fill 
```
All constructor arguments to `o3.TensorProduct` will work identically with
`TPProblem`; we support some additional arguments like the desired precision of
the weights and irreps. We recommend reading the [e3nn documentation and API
reference](https://docs.e3nn.org/en/latest/) first, then use our kernels 
as drop-in replacements. We support most "uvu" and "uvw" tensor products; 
see [this section](#tensor-products-we-support) for an up-to-date list of supported
configurations. 

**Important**: For many configurations, our codes return results identical to
e3nn up to floating point roundoff (in particular, all "uvu" problems with
multiplicity 1 for all irreps in the second input). For other configurations 
(e.g. "uvw"), we return identical results up to a well-defined reordering
of the weights relative to e3nn. 

## Installation 
We provide several options to build our package and replicate
the benchmarks in our preprint. Right now, we only support
source builds, but we provide scripts to streamline installation.

We highly recommend that you use
`conda` or `mamba` to set up a Python environment for installation.
You can build the package via `conda-build` or
`conda mambabuild`, or run `cmake` and `pip` directly if you prefer. 

### Build via conda or mambabuild for production (recommended)
1. **Setup**: Create a new conda environment, or activate an existing one.
You must install either `boa` or `conda-build`; we 
use `boa` for its speed. 
    ```bash
    shell> conda create --name my_env python=3.11 boa
    shell> conda activate 
    ``` 

2. **Install**: Clone, build, and install in three steps.
below: 
    ```bash
    (my_env) shell> git clone https://github.com/vbharadwaj-bk/equivariant_spmm.git
    (my_env) shell> conda mambabuild equivariant_spmm 
    (my_env) shell> conda install --use-local src 
    ```

    Use `build` in place of `mambabuild` if you
    installed `conda-build` in Step 1.

3. **Test**: You're ready to go!

### Build for development
You can also run `cmake` and use 
`pip` to install the package yourself, which can be useful
for development. We still recommend
you do this inside a `conda` environment, but you don't have to. Without
a `conda` environment, you become responsible for `cmake` and finding the right
path to NVIDIA's CUDA Toolkit. 

1. **Setup**: Create an environment with our core dependencies: 
    ```bash
    shell> conda create --name my_env python=3.11 Jinja2 pybind11 cmake numpy
    shell> conda activate 
    ``` 

2. **Install**: Build the C++ extension and install the package via `pip`: 
    ```bash
    (my_env) shell> cd equivariant_spmm/src 
    (my_env) shell> sh recipe/build.sh 
    (my_env) shell> pip install -e . -vv # Remove -e for non-editable install
    ``` 

### Build to replicate our benchmarks 
Follow either build process above. You'll also need the following packages: 
- `e3nn`, 
- `cuEquivariance`
- `cuEquivariance-torch` 
- `cuEquivariance-ops-torch-cu11` OR `cuEquivariance-ops-torch-cu12` 
- `matplotlib` (to reproduce our figures) 

We conducted our benchmarks on an NVIDIA A100-SXM-80GB GPU at
Lawrence Berkeley National Laboratory. Your results may differ 
a different GPU.

## Tensor products we accelerate 
e3nn supports a variety of connection modes for CG tensor products. We support accelerate
two that are commonly used in equivariant graph neural networks:
"uvu" and "uvw". Our JIT compiled kernels should handle:

1. Pure "uvu" tensor products, which are most efficient when the input with higher
multiplicities is the first argument. Our results are identical to e3nn when irreps in
the second input have multiplicity 1, and otherwise identical up to a reordering
of the input weights.

2. Pure "uvw" tensor products, which are currently more efficient when the input with
higher multiplicities is the first argument. Our results are identical to e3nn up to a reordering
of the input weights. 

Our code include correctness checks, but the configuration space is large. If you notice
a bug, let us know in a Github issue. We'll try our best to correct it or document the problem here.

### Lack of Support and Issues:
We do not yet support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".

If you have a use case for any of the unsupported features above, let us know. 
