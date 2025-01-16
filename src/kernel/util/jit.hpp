#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class __attribute__((visibility("default"))) JITKernel {
public:
   string gpu_program;

   JITKernel(ifstream& ifile); 
   JITKernel(string gpu_program);

   void compile(string kernel_name, const vector<int> template_params);
   void compile(vector<string> kernel_name, vector<vector<int>> template_params);

   void set_max_smem(int kernel_id, uint32_t max_smem_bytes);

   void execute(int kernel_id, uint32_t num_blocks, uint32_t num_threads, 
         void* args[], uint32_t smem=0, CUstream hStream=NULL);

   ~JITKernel();
private:
   nvrtcProgram prog;

   bool compiled = false;
   char* code = nullptr;

   CUdevice dev;
   CUlibrary library;

   vector<string> kernel_names;
   vector<CUkernel> kernels;
};

class __attribute__((visibility("default"))) KernelLaunchConfig {
public:
   uint32_t num_blocks = 0;
   uint32_t num_threads = 0;
   uint32_t warp_size = 32;
   uint32_t smem = 0; 

   KernelLaunchConfig() = default;
   ~KernelLaunchConfig() = default;
};

void test_jit();
