find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

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

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(espmm PUBLIC 
  ${pybind11_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/tensorproducts
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution
  )

set_target_properties(espmm PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(espmm PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)

# ------------------------------------------------------------
# Wrapper module
# ------------------------------------------------------------

# Manually build the module to avoid SOABI renaming. Original command:
#pybind11_add_module(kernel_wrapper kernel_wrapper.cpp)
#target_link_libraries(kernel_wrapper PRIVATE espmm)

add_library(kernel_wrapper MODULE kernel_wrapper.cpp)
target_link_libraries(kernel_wrapper PRIVATE pybind11::pybind11 espmm)
#target_link_options(kernel_wrapper PRIVATE -Wl,-rpath=${CMAKE_INSTALL_PREFIX})

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(kernel_wrapper)
endif()

set_target_properties(kernel_wrapper PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden"
                                         PREFIX "")

install(TARGETS espmm kernel_wrapper DESTINATION .)
