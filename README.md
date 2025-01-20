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

### Build via install script and pip (fastest) 
The steps below assume that you're using a bash shell and have a C / C++ 
compiler that CMake can find. If not, you can install [gxx](https://anaconda.org/conda-forge/gxx/) from `conda-forge`. 

1. **Setup**: Create an environment (or activate an existing one) with 
  our core dependencies: 
    ```bash
    shell> conda create -c conda-forge --name my_env python=3.11 pybind11 cmake nvidia::cuda-toolkit
    shell> conda activate 
    ``` 

2. **Install**: Build our package and install via `pip`: 
    ```bash
    cd equivariant_spmm
    sh dev_build.sh 
    pip install . # Use pip install -e . for an editable install 
    ``` 

3. **Test**: You're ready to go!

You don't have to install NVIDIA's CUDA toolkit or CMake if they exist on your
platform, but you're responsible for setting LD_LIBRARY_PATH so that libraries
are findable at runtime. Installing the CUDA toolkit via `conda` takes care of this for
you. 

### Build via conda or mambabuild
You can can also build our package via `conda-build` or
`conda mambabuild`. This can be much slower, but may help if you
encounter problems with the workflow above.

1. **Setup**: Create a new conda environment, or activate an existing one.
    You must install either `boa` or `conda-build`; we 
    use `boa` for its speed. 
    ```bash
    shell> conda create --name my_env python=3.11 conda-forge::boa mamba
    shell> conda activate my_env 
    ``` 

2. **Install**: Clone, build, and install in three steps:
    ```bash
    git clone https://github.com/vbharadwaj-bk/equivariant_spmm.git
    conda mambabuild ./equivariant_spmm 
    mamba install --use-local fast_tp 
    ```

    Use `build` and `conda` in place of `mambabuild` and `mamba`, 
    respectively, if you installed `conda-build` in Step 1.

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

We do not yet support:

- Mixing different instruction types in the same tensor product. 
- Instruction types besides "uvu" and "uvw".

If you have a use case for any of the unsupported features above, let us know. 
