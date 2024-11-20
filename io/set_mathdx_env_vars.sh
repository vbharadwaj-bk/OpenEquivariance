# These env vars came from: 
# https://docs.nersc.gov/development/build-tools/cmake/#linking-cuda-math-libraries

module load cudatoolkit

export CMAKE_PREFIX_PATH=${CUDA_HOME}/../../math_libs:${CMAKE_PREFIX_PATH}
export CPATH=${CUDA_HOME}/../../math_libs/include:${CPATH}
