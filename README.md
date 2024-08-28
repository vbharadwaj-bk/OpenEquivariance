# ESPMM

# Prerequisites
You will need the `cppimport` module, but the latest one in the Github 
repo for Perlmutter DVS, not the last release. Install it with

```bash
git clone https://github.com/tbenthompson/cppimport 
cd cppimport
python setup.py install
```

# Building the Code 

This repository contains two components: a C++ component
that is compiled / built with CMake, and a Python
wrapper that interfaces with the C++ layer through 
Pybind11. 

1. To set up the build, use the cmake configuration script:

```bash
vbharadw@login15> . cmake_configure.sh
```

2. You need to recompile the C++ layer every time you
make a change to a C++ source file OR header: 

```bash
vbharadw@login15> . compile.sh 
```

3. You can now run the driver classes at the Python level:

```bash
vbharadw@login15> . get_gpu_node.sh 
vbharadw@nid200264> . env.sh 
(nersc-python) vbharadw@nid200264> python driver.py
```
You only need to source the environment file `env.sh`
once per session, which activates the NERSC Python module. 
The script may trigger another round of building / linking against
the C++ library compiled earlier. You can now run the tests. 
