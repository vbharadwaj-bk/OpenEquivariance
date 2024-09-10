#include "jit.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

#define NVRTC_SAFE_CALL(x)                                      \
do {                                                            \
   nvrtcResult result = x;                                      \
   if (result != NVRTC_SUCCESS) {                               \
      std::cerr << "\nerror: " #x " failed with error "         \
               << nvrtcGetErrorString(result) << '\n';          \
      exit(1);                                                  \
   }                                                            \
} while(0)

JITKernel::JITKernel(ifstream& ifile)
: JITKernel::JITKernel(file_to_string(ifile))
{ }

JITKernel::JITKernel(string gpu_program) :
    gpu_program(gpu_program) {

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
    nvrtcCreateProgram( &prog,                // prog
                        gpu_program.c_str(),  // buffer
                        "kernel.cu",          // name
                        0,                    // numHeaders
                        NULL,                 // headers
                        NULL));               // includeNames 
}

~JITKernel::JITKernel {
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

void JITKernel::compile(string kernel_name, vector<int> template_params) {
    if(template_params.size() == 0) {
        kernel_names
    } 
}

string file_to_string(ifstream &ifile) {
    ifile.clear();
    ifile.seekg(0);

    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}
