#pragma once

class ESPMM_Context {
public:
    uint64_t node_count;
    uint64_t L1;
    uint64_t L2;
    uint64_t L3;

    ESPMM_Context(
        uint64_t node_count_i,
        uint64_t L1_i, 
        uint64_t L2_i, 
        uint64_t L3_i) :
        node_count(node_count_i),
        L1(L1_i), L2(L2_i), L3(L3_i) { }
};

void equivariant_spmm_cpu(
        ESPMM_Context &context,
        uint64_t edge_count,
        uint64_t* rows,
        uint64_t* cols,
        double* X_in,
        double* X_out,
        double* edge_features);

