# ESPMM
This repository contains a set of accelerated tensor product 
kernels for equivariant deep neural networks. It implements
a subset of the functionality of [e3nn](https://e3nn.org/)
for its common use cases in equivariant graph neural networks
(e.g. [Nequip](https://github.com/mir-group/nequip) or
[MACE](https://github.com/ACEsuit/mace)). 

We can provide an order of magnitude acceleration over e3nn
and around 2x speedup over 
[NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance),
which has a closed-source kernel package. We also offer fused
equivariant graph convolutions that can reduce memory consumption 
by multiple orders of magnitude.

We currently support NVIDIA GPUs; HIP support for AMD is planned! 

## Show me an example 
Here's a tensor product that appears in MACE, implemented in
e3nn: 

```python
To fill by Austin
```

And here's our code:

```python
To fill by Austin
```

If you're performing a tensor product as part of a graph 
convolution, you can fuse the two operations together to reduce both memory and compute time: 

```
```


## Building our Code
We provide several options to build our code and replicate
the benchmarks in our preprint; right now, we only support
source builds, but we provide scripts to streamline installation.

We highly recommend that you use
`conda` or `mamba` to set up a Python environment for installation.
You can build the package `conda-build` or
`conda mambabuild`, or run `cmake` and `pip` directly if you prefer. 

### Build via conda or mambabuild for production
1. **Setup**: Create a new conda environment, or activate an existing one.
You must install either `boa` or `conda-build`; we 
use `boa` for its speed. 
    ```bash
    shell> conda create --name my_env python=3.11 boa
    shell> conda activate 
    ``` 

2. **Install**: Clone our repo, build, and install the package: 
    ```bash
    (my_env) shell> git clone https://github.com/vbharadwaj-bk/equivariant_spmm.git
    (my_env) shell> conda mambabuild equivariant_spmm 
    (my_env) shell> conda install --use-local src 
    ```

    Use `build` in place of `mambabuild` if you
    installed `conda-build` in Step 1.

3. **Test**: You're ready to go!

### Build for development
You can also run `cmake` and create a local
`pip` installation, which can be useful
for development. 
