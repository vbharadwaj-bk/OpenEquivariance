#include "cutlass/gemm/warp/default_mma_tensor_op.h"

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

__global__ void warp_matmul(float* A, float* B, float* C) {
    int const kGemmK = 32;

    __shared__ cutlass::half_t smem_buffer_A[WarpMma::Shape::kM * kGemmK];
    __shared__ cutlass::half_t smem_buffer_B[WarpMma::Shape::kN * kGemmK];

    #pragma unroll
    for(int i = 0; i < WarpMma::Shape::kM * kGemmK; i += kGemmK) {
        int const k = i + threadIdx.x;
        if(k < WarpMma::Shape::kM) {
            smem_buffer_A[k] = A[k];
        }
    }

    #pragma unroll
    for(int i = 0; i < WarpMma::Shape::kN * kGemmK; i += kGemmK) {
        int const k = i + threadIdx.x;
        if(k < WarpMma::Shape::kN) {
            smem_buffer_B[k] = B[k];
        }
    }

    __syncwarp();

    //int t_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int lda = WarpMma::Shape::kM;
    int ldb = WarpMma::Shape::kN;

    WarpMma::IteratorA iter_A({smem_buffer_A, lda}, 0);
    WarpMma::IteratorB iter_B({smem_buffer_B, ldb}, 0);

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