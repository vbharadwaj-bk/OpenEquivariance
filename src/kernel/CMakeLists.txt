find_package(pybind11)
find_package(Python COMPONENTS Interpreter Development)

# Add library
add_library(espmm SHARED 
  espmm.hpp
  
  util/gpu_util.hpp
  util/buffer.hpp
  util/buffer.cpp
  util/jit.hpp
  util/jit.cpp
  util/device_prop.hpp
  util/device_prop.cpp
  util/mm_tester.hpp

  tensorproducts/tensorproducts.hpp
  tensorproducts/jit_tp.cpp

  convolution/convolution.hpp
  convolution/jit_conv.cpp
)

# target_link_libraries(espmm PUBLIC pybind11::headers)

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

#find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS 
# "${CMAKE_CURRENT_SOURCE_DIR}/../../nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/"
#)
#target_link_libraries(espmm PUBLIC mathdx::cublasdx)

target_include_directories(espmm PUBLIC 
  ${pybind11_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/tensorproducts
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution
  )

# Set target properties
set_target_properties(espmm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(espmm PROPERTIES LINKER_LANGUAGE CXX)

# To-do: should make path to standard CUDA math libraries a parameter 
target_link_directories(espmm PUBLIC /opt/nvidia/hpc_sdk/Linux_x86_64/2024/math_libs/12.4/lib64/)
target_link_libraries(espmm PUBLIC CUDA::cudart -lcublasLt CUDA::nvrtc)

install(TARGETS espmm )