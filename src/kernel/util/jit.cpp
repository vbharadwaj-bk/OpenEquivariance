#include "jit.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

/*
* This page is a useful resource to understand this file: 
* https://docs.nvidia.com/cuda/nvrtc/index.html#example-using-nvrtcgettypename
*/

#define NVRTC_SAFE_CALL(x)                                      \
do {                                                            \
   nvrtcResult result = x;                                      \
   if (result != NVRTC_SUCCESS) {                               \
      std::cerr << "\nerror: " #x " failed with error "         \
               << nvrtcGetErrorString(result) << '\n';          \
      exit(1);                                                  \
   }                                                            \
} while(0)

#define CUDA_SAFE_CALL(x)                                       \
do {                                                            \
   CUresult result = x;                                         \
   if (result != CUDA_SUCCESS) {                                \
      const char *msg;                                          \
      cuGetErrorName(result, &msg);                             \
      std::cerr << "\nerror: " #x " failed with error "         \
               << msg << '\n';                                  \
      exit(1);                                                  \
   }                                                            \
} while(0)

string file_to_string(ifstream &ifile) {
    ifile.clear();
    ifile.seekg(0);

    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

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

void JITKernel::compile(string kernel_name, vector<int> &template_params) {
    // Step 1: Generate kernel names from the template parameters 
    if(template_params.size() == 0) {
        kernel_names.push_back(kernel_name);
    }
    else {
        std::string result;
        for(int i = 0; i < template_params.size(); i++) {
            result += std::to_string(template_params[i]); 
            if(i != template_params.size() - 1) {
                result += ",";
            }
        }
        kernel_names.push_back(result);
    }

    // =========================================================
    // Step 2: Add name expressions, compile 
    for (size_t i = 0; i < name_vec.size(); ++i)
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_names[i].c_str()));

    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                    0,     // numOptions
                                                    NULL); // options

    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

    if (compileResult != NVRTC_SUCCESS) {
        throw std::logic_error("NVRTC Fail, log: " + std::string(log));
    } 
    delete[] log;
    compiled = true

    // =========================================================
    // Step 3: Get PTX, initialize device, context, and module 

    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

    for (size_t i = 0; i < kernel_names.size(); i++) {
        const char *name;

        NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                                prog,
                name_vec[i].c_str(), // name expression
                &name                // lowered name
                                            ));

        kernel.emplace_back();
        CUDA_SAFE_CALL(cuModuleGetFunction(&(kernels[i]), module, name));
    }
}

~JITKernel::JITKernel {
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    delete[] ptx;
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}
