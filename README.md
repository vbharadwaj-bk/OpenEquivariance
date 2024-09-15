# ESPMM

# Prerequisites
Before building this package, you must activate the Python environment 
that you want the resulting extension to use. At the
start of the configuration process, the command `which python` should
return the Python interpreter that you intend to use. On NERSC Perlmutter,
you can activate the nersc Python module by sourcing `.env.sh`.

You need Python3, CMake Version >=3.15, and CUDAToolkit>=12.2. We require the
following Python dependencies:

- numpy
- pybind11

# Building the Code 
The following steps build the C++ extension to Python using Pybind11.

1. To set up the build, use the cmake configuration script:

```bash
vbharadw@login15> . env.sh 
(nersc-python) vbharadw@login15> . cmake_configure.sh 
```

2. You need to recompile the C++ layer every time you
make a change to a C++ source file or header: 

```bash
(nersc-python) vbharadw@login15> . compile.sh 
```

3. You can now run the driver classes at the Python level. Remember to source
the environment script: 

```bash
vbharadw@login15> . get_gpu_node.sh 
vbharadw@nid200264> . env.sh 
(nersc-python) vbharadw@nid200264> python driver.py
```