#include "jit.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "gpu_util.hpp"

using namespace std;

/*
* This page is a useful resource on NVRTC: 
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

    if(ifile.fail()) {
        throw std::runtime_error("Failed to read kernel file.");
    }

    std::stringstream buffer;
    buffer << ifile.rdbuf();
    return buffer.str();
}

JITKernel::JITKernel(ifstream& ifile)
: JITKernel::JITKernel(file_to_string(ifile))
{ }

JITKernel::JITKernel(string gpu_program) :
    gpu_program(gpu_program) {

    gpuErrchk(cudaFree(0)); // No-op to initialize the primary context 
    NVRTC_SAFE_CALL(
    nvrtcCreateProgram( &prog,                // prog
                        gpu_program.c_str(),  // buffer
                        "kernel.cu",          // name
                        0,                    // numHeaders
                        NULL,                 // headers
                        NULL));               // includeNames
}

void JITKernel::compile(string kernel_name, const vector<int> template_params) {
    vector<string> kernel_names = {kernel_name};
    vector<vector<int>> template_param_list = {template_params};
    compile(kernel_names, template_param_list);
}

void JITKernel::compile(vector<string> kernel_names_i, vector<vector<int>> template_param_list) {
    if(compiled) {
        throw std::logic_error("JIT object has already been compiled!");
    }

    if(kernel_names_i.size() != template_param_list.size()) {
        throw std::logic_error("Kernel names and template parameters must have the same size!");
    }

    for(int kernel = 0; kernel < kernel_names_i.size(); kernel++) {
        string kernel_name = kernel_names_i[kernel];
        vector<int> &template_params = template_param_list[kernel];

        // Step 1: Generate kernel names from the template parameters 
        if(template_params.size() == 0) {
            kernel_names.push_back(kernel_name);
        }
        else {
            std::string result = kernel_name + "<";
            for(int i = 0; i < template_params.size(); i++) {
                result += std::to_string(template_params[i]); 
                if(i != template_params.size() - 1) {
                    result += ",";
                }
            }
            result += ">";
            kernel_names.push_back(result);
        }

    }
    
    // Prepare compilation options
    std::vector<const char*> opts = {
        "--std=c++17", // CublasDX had this one
        "--device-as-default-execution-space", // CublasDX had this one
        "-arch=sm_80",
        "--include-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.4/include/", // Add path to CUDA include directory
        "--include-path=/global/homes/a/aglover/equivariant_spmm/nvidia-mathdx-24.08.0/nvidia/mathdx/24.08",
        "--ptxas-options=-v", // This is just the verbose command
    };    

    // =========================================================
    // Step 2: Add name expressions, compile 
    for(size_t i = 0; i < kernel_names.size(); ++i)
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_names[i].c_str()));

    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                    static_cast<int>(opts.size()),     // numOptions
                                                    opts.data()); // options

    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

    if (compileResult != NVRTC_SUCCESS) {
        throw std::logic_error("NVRTC Fail, log: " + std::string(log));
    } 
    delete[] log;
    compiled = true;

    // =========================================================
    // Step 3: Get PTX, initialize device, context, and module 

    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));

    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    CUDA_SAFE_CALL(cuInit(0));


    // TODO: No context management here, we use the primary context 
    // CUdevice cuDevice;
    // CUcontext context;
    // CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    // CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

    for (size_t i = 0; i < kernel_names.size(); i++) {
        const char *name;

        NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                                prog,
                kernel_names[i].c_str(), // name expression
                &name                    // lowered name
                ));

        kernels.emplace_back();
        CUDA_SAFE_CALL(cuModuleGetFunction(&(kernels[i]), module, name));
    }


}

void JITKernel::set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
    if(kernel_id >= kernels.size())
        throw std::logic_error("Kernel index out of range!");

    cuFuncSetAttribute(kernels[kernel_id],
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    max_smem_bytes);
}

void JITKernel::execute(int kernel_id, uint32_t num_blocks, uint32_t num_threads, 
         void* args[], uint32_t smem, CUstream hStream) {

    if(kernel_id >= kernels.size())
        throw std::logic_error("Kernel index out of range!");

    CUDA_SAFE_CALL(
        cuLaunchKernel( kernels[kernel_id],
                        num_blocks, 1, 1,    // grid dim
                        num_threads, 1, 1,   // block dim
                        smem, hStream,       // shared mem and stream
                        args, NULL)            // arguments
    );            
}

JITKernel::~JITKernel() {
    if(compiled) {
        CUDA_SAFE_CALL(cuModuleUnload(module));
        delete[] ptx;
    }
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

// ========================= JIT TEST =========================

const char* JIT_TEST_CODE = R"(
using namespace std;

template<int template_int>
__global__ void f3(int test) {  
    printf("Hello, my argument is %d\n", test);
    printf("Hello, my template parameter is %d\n", template_int);
}

)";

void test_jit() {
    JITKernel jit(JIT_TEST_CODE);
    jit.compile("f3", { 3 });
    int test = 5;
    void *args[] = { &test };
    jit.execute(0, 1, 32, args);
}
