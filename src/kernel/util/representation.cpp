#include "representation.hpp"
#include "buffer.hpp"

#include <iostream>

using namespace std;
namespace py = pybind11;

void Representation::transpose_irreps_cpu(py::array_t<float> &rep_mat_py, bool row_major_in) {
    Buffer<float> rep_mat(rep_mat_py);
    vector<int> offsets = get_irrep_offsets();
    size_t rep_length = get_rep_length();

#pragma omp parallel
{
    Buffer<float> temp({rep_length});

    #pragma omp for 
    for(size_t i = 0; i < rep_mat.shape[0]; i++) {
        float* in_ptr = rep_mat.ptr + (i * rep_length);
        for(int j = 0; j < irreps.size(); j++) {
            float* irrep_start = in_ptr + offsets[j];
            float* temp_start = temp.ptr + offsets[j];

            int mult = get<0>(irreps[j]);
            int irrep_len = get<1>(irreps[j]) * 2 + 1;

            // Transpose irrep submatrix 
            for(int u = 0; u < irrep_len; u++) {
                for(int v = 0; v < mult; v++) {
                    if(row_major_in) {
                        temp_start[u * mult + v] = irrep_start[v * irrep_len + u];
                    }
                    else {
                        temp_start[v * irrep_len + u] = irrep_start[u * mult + v];
                    }
                }
            }
        } 
        std::copy(temp.ptr, temp.ptr + rep_length, in_ptr);
    }
}

}