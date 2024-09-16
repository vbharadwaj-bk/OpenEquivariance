#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class __attribute__((visibility("default"))) JITKernel {
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
