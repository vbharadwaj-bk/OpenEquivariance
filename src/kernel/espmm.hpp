#pragma once

// Taken from Stack Overflow
size_t round_up(size_t in, size_t multiple) {
    if (multiple == 0)
        return in;

    int remainder = in % multiple;
    if (remainder == 0)
        return in ;

    return in + multiple - remainder;
}  


struct ESPMM_Context {
public:
    // TODO: have this work for a sum of reps, not just one. 
    uint64_t node_count;
    uint64_t L1; // X_in representation
    uint64_t L2; // Edge feature representation
    uint64_t L3; // X_out representation

    size_t X_in_rowlen;
    size_t edge_rowlen;
    size_t X_out_rowlen; 

    ESPMM_Context() { };

    ESPMM_Context(
        uint64_t node_count_i,
        uint64_t L1_i, 
        uint64_t L2_i, 
        uint64_t L3_i) :
        node_count(node_count_i),
        L1(L1_i), L2(L2_i), L3(L3_i),
        X_in_rowlen(round_up(L1 * 2 + 1, 128 / sizeof(float))),
        edge_rowlen(round_up(L2 * 2 + 1, 128 / sizeof(float))),
        X_out_rowlen(round_up(L3 * 2 + 1, 128 / sizeof(float)))
        { }

    size_t get_X_in_rowlen() {
        return X_in_rowlen;
    }

    size_t get_edge_rowlen() {
        return edge_rowlen;
    }

    size_t get_X_out_rowlen() {
        return X_out_rowlen;
    }
};

void equivariant_spmm_cpu(
        ESPMM_Context &context,
        uint64_t edge_count,
        uint64_t* rows,
        uint64_t* cols,
        float* X_in,
        float* X_out,
        float* edge_features);

