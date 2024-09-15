find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(kernel_wrapper kernel_wrapper.cpp)

target_link_libraries(kernel_wrapper PRIVATE espmm)
target_link_options(kernel_wrapper PRIVATE -Wl,-rpath=${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS kernel_wrapper DESTINATION .)