find_package(pybind11 CONFIG REQUIRED)

# Manually build the module to avoid SOABI renaming. Original command:
#pybind11_add_module(kernel_wrapper kernel_wrapper.cpp)
#target_link_libraries(kernel_wrapper PRIVATE espmm)

add_library(kernel_wrapper MODULE kernel_wrapper.cpp)
target_link_libraries(kernel_wrapper PRIVATE pybind11::pybind11 espmm)
target_link_options(kernel_wrapper PRIVATE -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/lib)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(kernel_wrapper)
endif()

set_target_properties(kernel_wrapper PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden"
                                         PREFIX "")

install(TARGETS kernel_wrapper DESTINATION ../src)