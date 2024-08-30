# Add library
add_library(espmm SHARED 
  espmm.hpp
  espmm.cpp
  gpu_util.hpp
  buffer.hpp
  buffer.cpp
  thread_tensor_product.cu
  equivariant_convolution.cu
)

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(ESPMM_PUBLIC_HEADERS
    "espmm.hpp"
    "buffer.hpp"
)

# Set target properties
set_target_properties(espmm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(espmm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(espmm PROPERTIES PUBLIC_HEADER "${ESPMM_PUBLIC_HEADERS}")

# Link against CUDA
target_link_libraries(espmm ${CUDA_LIBRARIES})
install(TARGETS espmm PUBLIC_HEADER)
