//=========================================================================
/*
* Simple implementation that assigns one warp per nonzero and
* executes atomicAdd operations to accumulate to the output buffer.
*/
class __attribute__ ((visibility ("default"))) AtomicConvImpl  : public ConvolutionImpl {
public:
    AtomicConvImpl(RepTriple &io_reps) :
        ConvolutionImpl(io_reps) { };

    void exec_conv(
            float* L1_in,
            float* L2_in,
            float* L3_out,
            uint32_t* rows,
            uint32_t* cols,
            uint64_t nnz,
            uint32_t node_count,
            bool disable_tensor_op
            );

    ~AtomicConvImpl() = default;
};

//=========================================================================
/*
* Convolution implementation that uses shared memory to stage intermediates before
* writing out to global memory. 
* 
*/
class __attribute__ ((visibility ("default"))) SMConvImpl : public ConvolutionImpl {
public:
    SMConvImpl(RepTriple &io_reps) :
        ConvolutionImpl(io_reps) { };

    void exec_conv(
            float* L1_in,
            float* L2_in,
            float* L3_out,
            uint32_t* rows,
            uint32_t* cols,
            uint64_t nnz,
            uint32_t node_count,
            bool disable_tensor_op
            );

    ~SMConvImpl() = default;
};