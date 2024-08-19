#pragma once
#include <stdexcept>
//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>

// Taken from Stack Overflow
size_t round_up(size_t in, size_t multiple) {
    if (multiple == 0)
        return in;

    int remainder = in % multiple;
    if (remainder == 0)
        return in ;

    return in + multiple - remainder;
}  

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

class TensorProduct {
public:
    uint64_t L1; // X_in representation
    uint64_t L2; // Edge feature representation
    uint64_t L3; // X_out representation

    size_t L1_rowlen;
    size_t L2_rowlen;
    size_t L3_rowlen;

    TensorProduct(
        uint64_t L1_i, 
        uint64_t L2_i, 
        uint64_t L3_i) :
        L1(L1_i), L2(L2_i), L3(L3_i),
        L1_rowlen(round_up(L1 * 2 + 1, 128 / sizeof(float))),
        L2_rowlen(round_up(L2 * 2 + 1, 128 / sizeof(float))),
        L3_rowlen(round_up(L3 * 2 + 1, 128 / sizeof(float)))
        { }

    size_t get_row_length(int mode) {
        switch(mode) {
            case 1:
                return L1_rowlen;
            case 2:
                return L2_rowlen;
            case 3:
                return L3_rowlen;
            default:
                throw std::invalid_argument( "Invalid mode!" );

        }
    }

    virtual void exec_tensor_product_cpu(
            uint64_t num_products,
            float* X_in,
            float* X_out,
            float* edge_features) = 0;
};

/*
* A simple implementation that gets each thread 
* to handle each tensor product based on a coordinate format. 
*/
class ThreadTensorProduct : public TensorProduct {
    using TensorProduct::TensorProduct;

    void exec_tensor_product_cpu(
            uint64_t num_products,
            float* X_in,
            float* X_out,
            float* edge_features) {
        
        cout << "Executed successfully!" << endl;

    } 
};

