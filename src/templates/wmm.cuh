#include "cutlass/gemm/warp/default_mma_tensor_op.h"

#define ROW_OPERATION(ROW_LEN, LOOP_VAR, ...) \
    _Pragma ("unroll") \
    for(int LOOP_VAR = 0; LOOP_VAR < ROW_LEN; LOOP_VAR += THREADS_PER_WARP) { \
        if(LOOP_VAR >= ROW_LEN - THREADS_PER_WARP) { \
            if(lane_id < ROW_LEN - LOOP_VAR) { \
                __VA_ARGS__  \
            } \
        } \
        else { \
            __VA_ARGS__  \
        } \
    }


using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<cutlass::half_t>::value, 64>;

using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<cutlass::half_t>::value, 64>;

using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    cutlass::gemm::GemmShape<64, 64, 8>,                            // Overall warp-level GEMM operation
    cutlass::gemm::GemmShape<16, 8, 8>,                             // Target instruction
    cutlass::half_t, LayoutA,                                       // operand A type and layout
    cutlass::half_t, LayoutB,                                       // operand B type and layout
    float,                                                          // accumulator type
    cutlass::layout::RowMajor>::Type;                               // accumulator layout

__global__ void warp_matmul(cutlass::half_t *A, cutlass::half_t *B, float* C) {
    int const kGemmK = 64;
    int lane_id = threadIdx.x;
    int THREADS_PER_WARP = 32;

    __shared__ cutlass::half_t smem_buffer_A[WarpMma::Shape::kM * kGemmK];
    __shared__ cutlass::half_t smem_buffer_B[WarpMma::Shape::kN * kGemmK];
    __shared__ float smem_buffer_C[WarpMma::Shape::kM * WarpMma::Shape::kN];

    ROW_OPERATION(WarpMma::Shape::kM * kGemmK, i, smem_buffer_A[i + lane_id] = A[i + lane_id];)
    ROW_OPERATION(WarpMma::Shape::kN * kGemmK, i, smem_buffer_B[i + lane_id] = B[i + lane_id];)

    __syncwarp();

    //int t_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int lda = WarpMma::Shape::kM;
    int ldb = WarpMma::Shape::kN;
    int ldc = WarpMma::Shape::kN;

    WarpMma::IteratorA iter_A({smem_buffer_A, lda}, lane_id);
    WarpMma::IteratorB iter_B({smem_buffer_B, ldb}, lane_id);
    WarpMma::IteratorC iter_C({smem_buffer_C, ldc}, lane_id);

    WarpMma::FragmentA frag_A;
    WarpMma::FragmentB frag_B;
    WarpMma::FragmentC accum;

    WarpMma mma;
    accum.clear();

    #pragma unroll 1
    for (int k = 0; k < kGemmK; k += WarpMma::Shape::kK) { 
        iter_A.load(frag_A);  // Load fragments from A and B matrices
        iter_B.load(frag_B);

        ++iter_A; ++iter_B;   // Advance along GEMM K to next tile in A
                                //   and B matrices

                                // Compute matrix product
        mma(accum, frag_A, frag_B, accum);
    }

    __syncwarp();
    iter_C.store(accum);

    __syncwarp();
    ROW_OPERATION(WarpMma::Shape::kM * WarpMma::Shape::kN, i, C[i + lane_id] = smem_buffer_C[i + lane_id];)

}

/* Old implementation 

int t_idx = threadIdx.x + blockIdx.x * blockDim.x;

if(t_idx == 0) {
    for(int i = 0; i < {{M}}; i++) {
        for(int j = 0; j < {{N}}; j++) {
            float sum = 0;
            for(int k = 0; k < {{K}}; k++) {
                sum += A[i * {{K}} + k] * B[k * {{N}} + j];
            }
            C[i * {{N}} + j] = sum;
        }
    }
}

*/