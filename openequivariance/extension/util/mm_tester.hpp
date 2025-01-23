#pragma once

#include "jit.hpp"

class MMTester {
    JITKernel jit;

public:
    MMTester(std::string jit_kernel) : jit(jit_kernel) { 
        vector<string> kernels = {"warp_matmul"};
        jit.compile(kernels, {{}}); 
    }

    // Executes matmul with a single warp
    void execute(uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
        void* A_cast = reinterpret_cast<void*>(A_ptr);
        void* B_cast = reinterpret_cast<void*>(B_ptr);
        void* C_cast = reinterpret_cast<void*>(C_ptr);
        void *args[] = { &A_cast, &B_cast, &C_cast}; 
        jit.execute(0, 1, 32, args, 0);
    }
};