module load conda 
conda activate equivariant_spmm_env

module load cudatoolkit

export CMAKE_PREFIX_PATH=${CUDA_HOME}/../../math_libs:${CMAKE_PREFIX_PATH}
export CPATH=${CUDA_HOME}/../../math_libs/include:${CPATH}

# dcgmi profile --pause
# ncu -c 2 -o report.ncu-rep python driver.py
# dcgmi profile --resume 