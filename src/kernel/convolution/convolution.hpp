#pragma once

#include "util.hpp"

/*
* Graph convolution that combines node / edge features 
*/

class ESPMM_Context {  
public:
    // TODO: have this work for a sum of reps, not just one. 
    uint64_t node_count;
    uint64_t L1; // X_in representation
    uint64_t L2; // Edge feature representation
    uint64_t L3; // X_out representation

    size_t X_in_rowlen;
    size_t edge_rowlen;
    size_t X_out_rowlen; 

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