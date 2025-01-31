#pragma once

#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "jit.hpp"

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

class CUDA_Allocator {
    void* gpu_alloc (size_t size) {
        void* ptr;
        CUDA_CHECK( cudaMalloc((void**) &ptr, size ))
        return ptr;
    }

    void gpu_free (void* ptr) {
        CUDA_CHECK( cudaFree(ptr))
    }

    void copy_host_to_device (void* host, void* device, size_t size) {
        CUDA_CHECK( cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }

    void copy_device_to_host (void* host, void* device, size_t size) {
        CUDA_CHECK( cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }
}

class CUDATimer {
    cudaEvent_t start_evt, stop_evt;

public:
    GPUTimer() {  
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }

    void start() {
        cudaEventRecord(start_evt);
    }

    float stop_clock_get_elapsed() {
        float time_millis;
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time_millis, start_evt, stop_evt);
        return time_millis; 
    }

    void clear_L2_cache() {
        size_t element_count = 25000000;

        int* ptr = (int*) gpu_alloc (element_count * sizeof(int)) {
        CUDA_CHECK(cudaMemset(ptr, 42, element_count * sizeof(int)))
        gpu_free(ptr);
        cudaDeviceSynchronize();
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
};

class __attribute__((visibility("default"))) DeviceProp {
public:
    std::string name; 
    int warpsize;
    int major, minor;
    int multiprocessorCount;
    int maxSharedMemPerBlock;
    int maxSharedMemoryPerMultiprocessor; 

    DeviceProp(int device_id) {
        cudaDeviceProp prop; 
        cudaGetDeviceProperties(&prop, device_id);
        name = std::string(prop.name);
        CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        CUDA_CHECK(cudaDeviceGetAttribute(&warpsize, cudaDevAttrWarpSize, device_id));
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device_id));
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
        CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    }
};

/*
* This page is a useful resource on NVRTC: 
* https://docs.nvidia.com/cuda/nvrtc/index.html#example-using-nvrtcgettypename
*/

class CUJITKernel {
private:
    nvrtcProgram prog;

    bool compiled = false;
    char* code = nullptr;

    CUdevice dev;
    CUlibrary library;

    vector<string> kernel_names;
    vector<CUkernel> kernels;

public:
    CUJITKernel(string gpu_program) :
        gpu_program(gpu_program) {

        CUDA_CHECK(cudaFree(0)); // No-op to initialize the primary context 
        NVRTC_SAFE_CALL(
        nvrtcCreateProgram( &prog,                // prog
                            gpu_program.c_str(),  // buffer
                            "kernel.cu",          // name
                            0,                    // numHeaders
                            NULL,                 // headers
                            NULL));               // includeNames
    }

    void compile(string kernel_name, const vector<int> template_params) {
        vector<string> kernel_names = {kernel_name};
        vector<vector<int>> template_param_list = {template_params};
        compile(kernel_names, template_param_list);
    }

    void compile(vector<string> kernel_names_i, vector<vector<int>> template_param_list) {
        if(compiled) {
            throw std::logic_error("JIT object has already been compiled!");
        }

        if(kernel_names_i.size() != template_param_list.size()) {
            throw std::logic_error("Kernel names and template parameters must have the same size!");
        }

        for(unsigned int kernel = 0; kernel < kernel_names_i.size(); kernel++) {
            string kernel_name = kernel_names_i[kernel];
            vector<int> &template_params = template_param_list[kernel];

            // Step 1: Generate kernel names from the template parameters 
            if(template_params.size() == 0) {
                kernel_names.push_back(kernel_name);
            }
            else {
                std::string result = kernel_name + "<";
                for(unsigned int i = 0; i < template_params.size(); i++) {
                    result += std::to_string(template_params[i]); 
                    if(i != template_params.size() - 1) {
                        result += ",";
                    }
                }
                result += ">";
                kernel_names.push_back(result);
            }

        }
        
        DeviceProp dp(0); // TODO: We only query the first device at the moment
        std::string sm = "-arch=sm_" + std::to_string(dp.major) + std::to_string(dp.minor);

        std::vector<const char*> opts = {
            "--std=c++17",
            sm.c_str(),
            "--split-compile=0",
            "--use_fast_math"
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

        size_t codeSize;
        NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
        code = new char[codeSize];
        NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code));

        CUDA_SAFE_CALL(cuInit(0));
        CUDA_SAFE_CALL(cuLibraryLoadData(&library, code, 0, 0, 0, 0, 0, 0));

        for (size_t i = 0; i < kernel_names.size(); i++) {
            const char *name;

            NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                                    prog,
                    kernel_names[i].c_str(), // name expression
                    &name                    // lowered name
                    ));

            kernels.emplace_back();
            CUDA_SAFE_CALL(cuLibraryGetKernel(&(kernels[i]), library, name));
        }

        CUDA_SAFE_CALL(cuDeviceGet(&dev, 0));
    }

    void set_max_smem(int kernel_id, uint32_t max_smem_bytes) {
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        CUDA_SAFE_CALL(cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                max_smem_bytes,
                kernels[kernel_id],
                dev));
    }

    void execute(int kernel_id, void* args[], KernelLaunchConfig config) {
        if(kernel_id >= kernels.size())
            throw std::logic_error("Kernel index out of range!");

        CUcontext pctx = NULL; 
        CUDA_SAFE_CALL(cuCtxGetCurrent(&pctx));

        if(pctx == NULL) {
            CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&pctx, dev));
            CUDA_SAFE_CALL(cuCtxSetCurrent(pctx));
        }

        CUDA_SAFE_CALL(
            cuLaunchKernel( (CUfunction) (kernels[kernel_id]),
                            config.num_blocks, 1, 1,    // grid dim
                            config.num_threads, 1, 1,   // block dim
                            config.smem, config.hStream,       // shared mem and stream
                            args, NULL)          // arguments
        );            
    }

    ~CUJITKernel() {
        if(compiled) {
            CUDA_SAFE_CALL(cuLibraryUnload(library));
            delete[] code;
        }
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    }
};

