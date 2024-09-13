#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

// These macros can only be used once per file

// Creates a string for a JIT-only kernel
#define JITONLY_CODE(func) \
    const char* JIT_CODE = #func;

// Expands to both a JIT code and the function signature. 
#define JITBOTH_CODE(func) \
    const char* JIT_CODE = #func; \
    func

class JITKernel {
public:
   JITKernel(ifstream& ifile); 
   JITKernel(string gpu_program);

   // In the future, could compile more than one combo of kernel names
   // and template parameters
   void compile(string kernel_name, const vector<int> &template_params);

   void execute(uint32_t num_blocks, uint32_t num_threads, 
         void* args[], uint32_t smem=0, CUstream hStream=NULL);

   ~JITKernel();

private:
   string gpu_program;
   nvrtcProgram prog;

   bool compiled = false;
   char* ptx = nullptr;

   CUdevice cuDevice;
   CUcontext context;
   CUmodule module;

   vector<string> kernel_names;
   vector<CUfunction> kernels;
};

void test_jit();
