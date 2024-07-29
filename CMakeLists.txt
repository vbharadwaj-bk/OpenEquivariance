cmake_minimum_required(VERSION 3.10)
project(MyCudaLib)

# Set CUDA version
set(CUDA_VERSION 11.0)

# Find CUDA
find_package(CUDA ${CUDA_VERSION} REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_70)

# Create src directory
add_subdirectory(src/)