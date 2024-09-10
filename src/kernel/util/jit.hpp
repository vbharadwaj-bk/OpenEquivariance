#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class JITKernel {
public:
   JITKernel(ifstream& ifile); 
   JITKernel(string gpu_program);

   // In the future, could compile more than one combo of kernel names
   // and template parameters
   void compile(string kernel_name, vector<int> &template_params);
   int execute();

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

