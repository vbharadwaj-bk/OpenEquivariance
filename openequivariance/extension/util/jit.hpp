#pragma once

#include <nvrtc.h>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class __attribute__((visibility("default"))) KernelLaunchConfig {
public:
   uint32_t num_blocks = 0;
   uint32_t num_threads = 0;
   uint32_t warp_size = 32;
   CUstream hStream = NULL;

   KernelLaunchConfig() = default;
   ~KernelLaunchConfig() = default;
};

/*
class __attribute__((visibility("default"))) JITKernel {
   void compile(string kernel_name, const vector<int> template_params) = 0;
   void compile(vector<string> kernel_name, vector<vector<int>> template_params) = 0;
   void set_max_smem(int kernel_id, uint32_t max_smem_bytes) = 0;
   void execute(int kernel_id, void* args[], KernelLaunchConfig config) = 0;
}
*/