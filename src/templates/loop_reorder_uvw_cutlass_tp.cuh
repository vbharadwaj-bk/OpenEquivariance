{# Jinja2 Tempate #}
constexpr int THREADS_PER_WARP = {{forward_config.warp_size}};

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier> 

#include <cute/tensor.hpp>
#include <cute/util/print.hpp>


{%- from 'macros.jinja' import declare_smem_arrays with context %}

#define DEBUG_PRINTING 0

namespace cg = cooperative_groups;
typedef cg::thread_block_tile<32> Warp_Tile;

// FORWARD DECLARATIONS
template<typename T, typename Group, int Num_Elements> 
__device__ __forceinline__ void static_contiguous_copy(Group, T*, T*);
template<typename T, typename Group, int Num_Elements> 
__device__ __forceinline__ void static_contiguous_set (Group, T*, T);
template<typename T, typename Group> 
__device__ __forceinline__ void dynamic_contiguous_copy(Group, T*, T*, int); 
template<typename T, typename Group>
__device__ __forceinline__ void dynamic_contiguous_set (Group, T*, T , int);
template<
    typename T, 
    typename Group, 
    int in1_extent, 
    int in2_extent, 
    int out_extent, 
    int in1_offset, 
    int in2_offset, 
    int out_offset,
    int in1_multiplicity, 
    int in2_multiplicity, 
    int out_multiplicity
    >
__device__ __forceinline__ void weight_view_condensing_copy(Group g, T* dst, T* src);
template<
    typename T, 
    typename Group, 
    int in1_extent, 
    int in2_extent, 
    int out_extent, 
    int in1_offset, 
    int in2_offset, 
    int out_offset,
    int in1_multiplicity, 
    int in2_multiplicity, 
    int out_multiplicity
    >
__device__ __forceinline__ void weight_view_expanding_global_atomic(Group g, T* dst, T* src);

template<typename Tensor>
__device__ __forceinline__ void print_tensor_contents(Tensor t); 

// Templated Forward Declarations
{%- for i in range(len(InstructionInfoList)) %}
template<typename Tile>
__device__ __forceinline__ void forward_kernel_shared_memory_instruction_{{i}}(Tile, float*, float*, float*, float*, float*, float*);
template<typename Tile> 
__device__ __forceinline__ void backward_kernel_shared_memory_instruction_{{i}}(Tile, float*, float*, float*, float*, float*, float*, float*, float*, float*);
{%- endfor %}


// Generic kernel which calls a subkernel for each forward interaction
// column-major data layout 
__global__ void forward(
    size_t num_products, 
    float* L1_in, 
    float* L2_in,
    float* L3_out,
    float* weights
    ) 
{   
    {%- set warps_per_block = divide(forward_config.num_threads, forward_config.warp_size) %}
    constexpr int warps_per_block = {{warps_per_block}}; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP; 
    // int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % {{warps_per_block}}; 
    
    {{ declare_smem_arrays( { 
        "common": [
            ("weights_smem", "float", max_weight_size),
            ], 
        "per_warp": [
            ("L1_smem_warp", "float", max_in1_instruction_size),
            ("L2_smem_warp", "float", max_in2_instruction_size), 
            ("L3_smem_warp", "float", max_out_instruction_size),
            ("gemm_L3_smem_warp", "float", forward_smem_gemm_info.L3_scratch_elems),
            ("gemm_weights_smem_warp", "float", forward_smem_gemm_info.weight_scratch_elems),
            ],
        }, "warp_loc", forward_config)}}    

    cg::thread_block block = cg::this_thread_block(); 
    Warp_Tile warp_tile = cg::tiled_partition<THREADS_PER_WARP>(block);
    
    
    {%- for kernel_ID, II in enumerate(InstructionInfoList) %}
    {   
        constexpr int L1_mults = {{II.in1_multiplicity}};
        constexpr int L2_mults = {{II.in2_multiplicity}};
        constexpr int L3_mults = {{II.out_multiplicity}};
        
        constexpr int L1_irrep_length = {{II.in1_irrep_length}}; 
        constexpr int L2_irrep_length = {{II.in2_irrep_length}};
        constexpr int L3_irrep_length = {{II.out_irrep_length}}; 

        constexpr int L1_size_instruction = L1_mults * L1_irrep_length;
        constexpr int L2_size_instruction = L2_mults * L2_irrep_length;
        constexpr int L3_size_instruction = L3_mults * L3_irrep_length;   

        constexpr int weights_size_instruction = {{product(II.path_shape)}}; //L1_mults * L2_mults * L3_mults; 
        
        block.sync(); 

        float* L1_global_shift_instruction = L1_in  + {{II.in1_offset}}; 
        float* L2_global_shift_instruction = L2_in  + {{II.in2_offset}}; 
        float* L3_global_shift_instruction = L3_out + {{II.out_offset}}; 

        float* weights_shift_instruction = weights + {{II.weight_offset}};

        dynamic_contiguous_copy<float, cg::thread_block>(block, weights_smem, weights_shift_instruction, weights_size_instruction);  

        block.sync(); 

        // GRID STRIDE LOOP
        for(
            int batch_index = (blockIdx.x * warps_per_block) + warp_loc; 
            batch_index < num_products; 
            batch_index += (gridDim.x * warps_per_block) 
            )
        {   
            // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
            float* L1_global_shift_warp = L1_global_shift_instruction + (batch_index * {{L1.rep_len}}); 
            float* L2_global_shift_warp = L2_global_shift_instruction + (batch_index * {{L2.rep_len}}); 
            float* L3_global_shift_warp = L3_global_shift_instruction + (batch_index * {{L3.rep_len}}); 

            // COPY FROM GLOBAL
            static_contiguous_copy<float, Warp_Tile, L1_size_instruction> (warp_tile, L1_smem_warp, L1_global_shift_warp);
            static_contiguous_copy<float, Warp_Tile, L2_size_instruction> (warp_tile, L2_smem_warp, L2_global_shift_warp);
            static_contiguous_copy<float, Warp_Tile, L3_size_instruction> (warp_tile, L3_smem_warp, L3_global_shift_warp); 
            warp_tile.sync(); 

            // PERFORM TP 
            forward_kernel_shared_memory_instruction_{{kernel_ID}}<Warp_Tile>(
                warp_tile, 
                L1_smem_warp, 
                L2_smem_warp, 
                gemm_L3_smem_warp,
                gemm_weights_smem_warp, 
                L3_smem_warp, 
                weights_smem
                );

            warp_tile.sync();
            
            // WRITE TO GLOBAL
            static_contiguous_copy<float, Warp_Tile, L3_size_instruction> (warp_tile, L3_global_shift_warp, L3_smem_warp);
        } 
    }
    {% endfor %}   
}

// Generic kernel which calls a subkernel for each backward interaction
// column-major data layout
/*
* Backward pass kernel. Currently assumes that each tensor product
* has a shared set of weights
* 
* Inputs:
*   L1_in, L2_in, weights, L3_grad
* Outputs:
*   L1_grad, L2_grad, weights_grad 
*/
__global__ void backward(
    size_t num_products,
    float* L1_in, 
    float* L1_grad,
    float* L2_in, 
    float* L2_grad, 
    float* weights, 
    float* weights_grad,
    float* L3_grad) 
{
    {%- set warps_per_block = divide(backward_config.num_threads, backward_config.warp_size) %}
    constexpr int warps_per_block = {{warps_per_block}}; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP; 
    // int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % warps_per_block; 

    {{ declare_smem_arrays( { 
        "common": [
            ("weights_smem", "float", max_weight_size),
            ("weights_grad_smem", "float", max_weight_size),
            ], 
        "per_warp": [
            ("L1_shared_shift_warp", "float", max_in1_instruction_size),
            ("L1_grad_shared_shift_warp", "float", max_in1_instruction_size), 
            ("L2_shared_shift_warp", "float", max_in2_instruction_size),
            ("L2_grad_shared_shift_warp", "float", max_in2_instruction_size), 
            ("L3_grad_shared_shift_warp", "float", max_out_instruction_size),
            ("gemm_L1L2_smem_warp", "float", backward_smem_gemm_info.L1L2_scratch_elems),
            ("gemm_weights_smem_warp", "float", backward_smem_gemm_info.weight_scratch_elems),             
        ]
        }, "warp_loc", backward_config)}}    

    cg::thread_block block = cg::this_thread_block();
    Warp_Tile warp_tile = cg::tiled_partition<THREADS_PER_WARP>(block);
    
    {%- for kernel_ID, II in enumerate(InstructionInfoList) %}
    {   
        constexpr int L1_mults = {{II.in1_multiplicity}};
        constexpr int L2_mults = {{II.in2_multiplicity}};
        constexpr int L3_mults = {{II.out_multiplicity}};
        
        constexpr int L1_irrep_length = {{II.in1_irrep_length}}; 
        constexpr int L2_irrep_length = {{II.in2_irrep_length}};
        constexpr int L3_irrep_length = {{II.out_irrep_length}}; 

        constexpr int L1_size_instruction = L1_mults * L1_irrep_length;
        constexpr int L2_size_instruction = L2_mults * L2_irrep_length;
        constexpr int L3_size_instruction = L3_mults * L3_irrep_length;   

        constexpr int in1_weight_extent = {{II.weight_in1_extent}}; 
        constexpr int in2_weight_extent = {{II.weight_in2_extent}};
        constexpr int out_weight_extent = {{II.weight_out_extent}}; 

        constexpr int in1_weight_offset = {{II.weight_in1_offset}};
        constexpr int in2_weight_offset = {{II.weight_in2_offset}};
        constexpr int out_weight_offset = {{II.weight_out_offset}};

        constexpr int weights_size_instruction = {{product(II.path_shape)}}; //L1_mults * L2_mults * L3_mults; 
        
        block.sync(); 

        float* L1_global_shift_instruction      = L1_in   + {{II.in1_offset}}; 
        float* L1_grad_global_shift_instruction = L1_grad + {{II.in1_offset}}; 
        float* L2_global_shift_instruction      = L2_in   + {{II.in2_offset}}; 
        float* L2_grad_global_shift_instruction = L2_grad + {{II.in2_offset}}; 
        float* L3_grad_global_shift_instruction = L3_grad + {{II.out_offset}}; 

        float* weights_shift_instruction = weights + {{II.weight_offset}};
        float* weights_grad_shift_instruction = weights_grad + {{II.weight_offset}}; 

        // dynamic_contiguous_copy<float, cg::thread_block>(block, weights_smem, weights_shift_instruction, weights_size_instruction);  

        if (warp_loc == 0){ 
            weight_view_condensing_copy<
                float,
                Warp_Tile,
                in1_weight_extent,
                in2_weight_extent,
                out_weight_extent, 
                in1_weight_offset,
                in2_weight_offset,
                out_weight_extent,
                L1_mults,
                L2_mults,
                L3_mults
            >(warp_tile, weights_smem, weights_shift_instruction);
        }
        block.sync(); 

        dynamic_contiguous_set<float, cg::thread_block>(block, weights_grad_smem, 0.0f, weights_size_instruction);  

        block.sync();
        // GRID STRIDE LOOP
        for(
            int batch_index = (blockIdx.x * warps_per_block) + warp_loc; 
            batch_index < num_products; 
            batch_index += (gridDim.x * warps_per_block)
            )
        {   
            // THIS FOR LOOP TAKES INTO ACCOUNT THAT EACH WARP IS DOING A DIFFERENT BATCH ELEMENT 

            // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
            float* L1_global_shift_warp         = L1_global_shift_instruction       + (batch_index * {{L1.rep_len}}); 
            float* L1_grad_global_shift_warp    = L1_grad_global_shift_instruction  + (batch_index * {{L1.rep_len}}); 
            float* L2_global_shift_warp         = L2_global_shift_instruction       + (batch_index * {{L2.rep_len}}); 
            float* L2_grad_global_shift_warp    = L2_grad_global_shift_instruction  + (batch_index * {{L2.rep_len}}); 
            float* L3_grad_global_shift_warp    = L3_grad_global_shift_instruction  + (batch_index * {{L3.rep_len}}); 
            
            // COPY FROM GLOBAL
            static_contiguous_copy<float, Warp_Tile, L1_size_instruction>(warp_tile, L1_shared_shift_warp, L1_global_shift_warp);
            static_contiguous_copy<float, Warp_Tile, L1_size_instruction>(warp_tile, L1_grad_shared_shift_warp, L1_grad_global_shift_warp);  
            static_contiguous_copy<float, Warp_Tile, L2_size_instruction>(warp_tile, L2_shared_shift_warp, L2_global_shift_warp);
            static_contiguous_copy<float, Warp_Tile, L2_size_instruction>(warp_tile, L2_grad_shared_shift_warp, L2_grad_global_shift_warp);
            static_contiguous_copy<float, Warp_Tile, L3_size_instruction>(warp_tile, L3_grad_shared_shift_warp, L3_grad_global_shift_warp); 

            warp_tile.sync(); 

            // PERFORM TP 
            backward_kernel_shared_memory_instruction_{{kernel_ID}}<Warp_Tile>(
                warp_tile, 
                L1_shared_shift_warp, 
                L1_grad_shared_shift_warp,  
                L2_shared_shift_warp, 
                L2_grad_shared_shift_warp, 
                L3_grad_shared_shift_warp, 
                weights_smem, 
                weights_grad_smem,
                gemm_L1L2_smem_warp, 
                gemm_weights_smem_warp
                );

            warp_tile.sync();
            
            // WRITE TO GLOBAL
            static_contiguous_copy<float, Warp_Tile, L1_size_instruction>(warp_tile, L1_grad_global_shift_warp, L1_grad_shared_shift_warp);
            static_contiguous_copy<float, Warp_Tile, L2_size_instruction>(warp_tile, L2_grad_global_shift_warp, L2_grad_shared_shift_warp);
        }

        block.sync(); 

        // global atomics 
        // for(int weight_copy_index = block.thread_rank(); weight_copy_index < weights_size_instruction; weight_copy_index += block.size()){
        //     atomicAdd(&weights_grad_shift_instruction[weight_copy_index], weights_grad_smem[weight_copy_index]); 
        // }
        if (warp_loc == 0){
        weight_view_expanding_global_atomic<
            float,
            Warp_Tile, 
            in1_weight_extent,
            in2_weight_extent,
            out_weight_extent, 
            in1_weight_offset,
            in2_weight_offset,
            out_weight_extent,
            L1_mults,
            L2_mults,
            L3_mults
         >(warp_tile, weights_grad_shift_instruction, weights_grad_smem); 
        }

        block.sync();
    }
    {% endfor %}   
}

// UTILITY FUNCTIONS HERE

template<
    typename T, 
    typename Group, 
    int in1_extent, 
    int in2_extent, 
    int out_extent, 
    int in1_offset, 
    int in2_offset, 
    int out_offset,
    int in1_multiplicity, 
    int in2_multiplicity, 
    int out_multiplicity
    >
__device__ __forceinline__ void weight_view_condensing_copy(Group g, T* dst, T* src){
    static_assert(in1_extent > 0);
    static_assert(in2_extent > 0);
    static_assert(out_extent > 0);
    static_assert(in1_multiplicity > 0);
    static_assert(in2_multiplicity > 0);
    static_assert(out_multiplicity > 0);
    static_assert(out_multiplicity <= g.size()); 

    bool active = g.thread_rank() < out_multiplicity; 
    int out_index = g.thread_rank();
    #pragma unroll 
    for(int in1_index = 0; in1_index < in1_multiplicity; in1_index++){
    #pragma unroll 
    for(int in2_index = 0; in2_index < in2_multiplicity; in2_index++){
          if (active){
            int src_weight_index = (
                    ((in1_index + in1_offset) * (in2_extent * out_extent)) 
                + ((in2_index + in2_offset) * (out_extent)) 
                + (out_index + out_offset)); 
            int dst_weight_index = (
                  ((in1_index) * (in2_multiplicity * out_multiplicity))  
                + ((in2_index) * (out_multiplicity)) 
                + ((out_index))
                );
            dst[dst_weight_index] = src[src_weight_index]; 
        }
    }
    }
}

template<
    typename T, 
    typename Group, 
    int in1_extent, 
    int in2_extent, 
    int out_extent, 
    int in1_offset, 
    int in2_offset, 
    int out_offset,
    int in1_multiplicity, 
    int in2_multiplicity, 
    int out_multiplicity
    >
__device__ __forceinline__ void weight_view_expanding_global_atomic(Group g, T* dst, T* src){
    static_assert(in1_extent > 0);
    static_assert(in2_extent > 0);
    static_assert(out_extent > 0);
    static_assert(in1_multiplicity > 0);
    static_assert(in2_multiplicity > 0);
    static_assert(out_multiplicity > 0);
    static_assert(out_multiplicity <= g.size()); 

    bool active = g.thread_rank() < out_multiplicity; 
    int out_index = g.thread_rank();
    #pragma unroll 
    for(int in1_index = 0; in1_index < in1_multiplicity; in1_index++){
    #pragma unroll 
    for(int in2_index = 0; in2_index < in2_multiplicity; in2_index++){
          if (active){
            int dst_weight_index = (
                    ((in1_index + in1_offset) * (in2_extent * out_extent)) 
                + ((in2_index + in2_offset) * (out_extent)) 
                + (out_index + out_offset)); 
            int src_weight_index = (
                  ((in1_index) * (in2_multiplicity * out_multiplicity))  
                + ((in2_index) * (out_multiplicity)) 
                + ((out_index))
                );
            atomicAdd(&dst[dst_weight_index], src[src_weight_index]); 
        }
    }
    }
}


template<typename T, typename Group, int Num_Elements> 
__device__ __forceinline__ void static_contiguous_copy(Group g, T* dst, T* src){
    static_assert(Num_Elements > 0);
    static_assert(g.size() > 0); 
    static_assert(g.size() <= 32);
    int thread_rank = g.thread_rank();


    #pragma unroll
    for(int i = 0; i + g.size() <= Num_Elements; i += g.size()){
        dst[i + thread_rank] = src[i + thread_rank];
    }
    
    constexpr int remainder = Num_Elements % g.size(); 
    if constexpr (remainder != 0){
        constexpr int quotient = Num_Elements / g.size();
        constexpr int offset = quotient * g.size();  
        if (thread_rank < remainder){
            dst[offset + thread_rank] = src[offset + thread_rank];
        }
    }
}


template<typename T, typename Group, int Num_Elements> 
__device__ __forceinline__ void static_contiguous_set(Group g, T* dst, T value){
    static_assert(Num_Elements > 0);
    static_assert(g.size() > 0); 
    static_assert(g.size() <= 32);
    int thread_rank = g.thread_rank();

    #pragma unroll
    for(int i = 0; i + g.size() <= Num_Elements; i += g.size()){
        dst[i + thread_rank] = value;
    }
    
    constexpr int remainder = Num_Elements % g.size(); 
    if constexpr (remainder != 0){
        constexpr int quotient = Num_Elements / g.size();
        constexpr int offset = quotient * g.size();  
        if (thread_rank < remainder){
            dst[offset + thread_rank] = value;
        }
    }
}

template<typename T, typename Group> 
__device__ __forceinline__ void dynamic_contiguous_copy(Group g, T* dst, T* src, int n){
    int group_size = g.size(); 
    int thread_rank = g.thread_rank(); 
    for(int i = 0; i < n; i += group_size){
        if (i + thread_rank < n){
            dst[i + thread_rank] = src[i + thread_rank]; 
        }
    }
}

template<typename T, typename Group>
__device__ __forceinline__ void dynamic_contiguous_set(Group g, T* ptr, T value, int n){
    int group_size = g.size(); 
    int thread_lane = g.thread_rank(); 
    for(int i = 0; i < n; i+= group_size){
        if (i + thread_lane < n){
            ptr[i + thread_lane] = value; 
        }
    }
}

template<typename Tensor>
__device__ __forceinline__ void print_tensor_contents(Tensor t){
    for (int row = 0; row < cute::size<0>(t); row++){
        for (int col = 0; col < cute::size<1>(t); col++){
            cute::print("%+1.3f",t(row,col)); cute::print(" ");
        } 
        cute::print("\n");
    }
}
