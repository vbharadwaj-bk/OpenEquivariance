{# Jinja2 Tempate #}
constexpr int FORWARD_THREADS_PER_WARP = {{forward_config.warp_size}};
// constexpr int FORWARD_THREAD_BLOCK_SIZE = {{forward_config.num_threads}}; 
constexpr int FORWARD_THREADS_PER_BATCH_ELEMENT = 32; // HARDCODED NONSENSE
constexpr int FORWARD_BATCH_ELEMENTS_PER_THREAD_BLOCK = {{forward_config.num_threads}} / 32; // HARDCODED NONSENSE

// constexpr int BACKWARD_THREADS_PER_WARP = {{backward_config.warp_size}};
// constexpr int BACKWARD_THREAD_BLOCK_SIZE = {{backward_config.num_threads}};

//#include <cooperative_groups.h>

{%- from 'macros.jinja' import declare_smem_arrays with context %}

// FORWARD DECLARATIONS
__device__ __forceinline__ void contiguous_copy(float*, float* , int);
__device__ __forceinline__ void contiguous_set (float*, float  , int);

__device__ __forceinline__ void forward_kernel_shared_memory(
    const float* __restrict__ L1_smem, 
    const float* __restrict__ L2_smem, 
          float* __restrict__ L3_smem,
    const float* __restrict__ weights 
    ){ 
    float L1_local_vec[{{L1.irrep_lengths | max}}];
    float L2_local_vec[{{L2.irrep_lengths | max}}];
    float L3_local_vec[{{L3.irrep_lengths | max}}];
    float output_local_vec[{{L3.irrep_lengths | max}}];
    float local_weight; 

    {%- set num_interact = interactions | length %}
    {%- for interaction_index in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[interaction_index] %}
        {%- set weight_offset = weight_offsets[interaction_index] %}
        
        // shift weights 
        const float* weights_shft = weights + {{weight_offset}}; 

        // THREADS ONLY PERFORM CALCULATIONS IF THE THREAD ID IS RESPOSIBLE FOR ONE OUTPUT 
        if ((threadIdx.x % FORWARD_THREADS_PER_WARP) < {{L3.mults[w]}}){
            // CLEAR OUTPUTS
            #pragma unroll 
            for (int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[v]}}; L3_irrep_index++){
                output_local_vec[L3_irrep_index] = 0.0f; 
            }

            // ITERATE THROUGH L1_MULTs 
            #pragma unroll
            for (int L1_mult_index = 0; L1_mult_index < {{L1.mults[u]}}; L1_mult_index++){
                // LOAD L1_SMEM INTO REGISTERS 
                #pragma unroll
                for (int L1_irrep_index = 0; L1_irrep_index < {{L1.irrep_lengths[u]}}; L1_irrep_index++){
                    L1_local_vec[L1_irrep_index] = L1_smem[L1_irrep_index + (L1_mult_index * {{L1.irrep_lengths[u]}}) + {{L1.offsets[u]}}];  
                }
                // ITERATE THROUGH L2_MULTs
                #pragma unroll 
                for(int L2_mult_index = 0; L2_mult_index < {{L2.mults[w]}}; L2_mult_index++){
                    // READ WEIGHT FROM GLOBAL MEMORY
                    local_weight = weights[(L1_mult_index * {{L1.mults[u]}} * {{L2.mults[v]}}) + (L2_mult_index * {{L2.mults[w]}}) + (threadIdx.x % FORWARD_THREADS_PER_WARP)];

                    //LOAD L2_SMEM INTO REGISTERS
                    #pragma unroll 
                    for(int L2_irrep_index = 0; L2_irrep_index < {{L2.irrep_lengths[v]}}; L2_irrep_index++){
                        L2_local_vec[L2_irrep_index] = L1_smem[L2_irrep_index + (L2_mult_index * {{L2.irrep_lengths[v]}}) + {{L2.offsets[v]}}];
                    }

                    // CLEAR L3 REGISTERS
                    #pragma unroll
                    for(int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                        L3_local_vec[L3_irrep_index] = 0.0f;
                    }

                    // PERFORM CG DECOMPOSITION CALCULATION 
                    {%- for i in range(tensor.nnz) %}
                        {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                        L3_local_vec[{{coord3}}] += {{value}} * L1_local_vec[{{coord1}}] * L2_local_vec[{{coord2}}];
                    {%- endfor %}

                    // APPLY WEIGHT AND SUM TO OUTPUT
                    #pragma unroll 
                    for(int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                        output_local_vec[L3_irrep_index] += L3_local_vec[L3_irrep_index] * local_weight; 
                    }

                }
            }  

            // WRITE THE RESULTS OUT TO SHARED MEMORY
            #pragma unroll
            for (int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                L3_smem[((threadIdx.x % FORWARD_THREADS_PER_WARP) * {{L3.irrep_lengths[w]}}) + {{L3.offsets[w]}}] = output_local_vec[L3_irrep_index]; 
            }
        }  
    {%- endfor %}  
}

// Generic kernel which calls a subkernel for each forward interaction
// column-major data layout 
__global__ void forward(
    size_t num_products, 
    float* L1_in, 
    float* L2_in,
    float* weights, 
    float* L3_out
    ) 
{
    {%- set warps_per_block = divide(forward_config.num_threads, forward_config.warp_size) %}
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / FORWARD_THREADS_PER_WARP; 
    int lane_id = idx % FORWARD_THREADS_PER_WARP;
    int warp_loc = warp_id % {{warps_per_block}}; 
    // size_t warps_launched = blockDim.x * gridDim.x / FORWARD_THREADS_PER_WARP; 
    // size_t num_products_per_warp = (num_products + warps_launched - 1) / warps_launched; 

    {{ declare_smem_arrays( { 
        "common": [], 
        "per_warp": [
            ("L1_smem", "float", L1.rep_len),
            ("L2_smem", "float", L2.rep_len), 
            ("L3_smem", "float", L3.rep_len)
            ]}, "warp_loc", forward_config)}}    

    
    // GRID STRIDE LOOP
    for(
        int block_batch_element_index = blockIdx.x * FORWARD_BATCH_ELEMENTS_PER_THREAD_BLOCK; 
        block_batch_element_index < num_products; 
        block_batch_element_index += FORWARD_BATCH_ELEMENTS_PER_THREAD_BLOCK * gridDim.x
        )
    {   
        // CHECK IF THIS IS THE LAST BLOCK
        int block_batch_elements_to_process = min((int)(FORWARD_BATCH_ELEMENTS_PER_THREAD_BLOCK), (int)(num_products - block_batch_element_index)); 
        // CHECK WHICH BATCH ELEMENT A SPECIFIC THREAD WILL ACT UPON
        int thread_batch_element_index = block_batch_element_index + (lane_id / FORWARD_THREADS_PER_BATCH_ELEMENT);

        // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
        float* L1_shift = L1_in  + (thread_batch_element_index * {{L1.rep_len}}); 
        float* L2_shift = L2_in  + (thread_batch_element_index * {{L2.rep_len}}); 
        float* L3_shift = L3_out + (thread_batch_element_index * {{L3.rep_len}}); 
        
        // COPY IN FROM GLOBAL, RESET SMEM
        contiguous_copy(L1_shift, L1_smem, block_batch_elements_to_process * {{L1.rep_len}}); 
        contiguous_copy(L2_shift, L2_smem, block_batch_elements_to_process * {{L2.rep_len}});
        contiguous_set ( L3_smem, 0.0f   , block_batch_elements_to_process * {{L3.rep_len}});         
        
        __syncthreads();
        // IF THE THREAD IS ASSOCIATED WITH A VALID BATCH INDEX
        if (thread_batch_element_index < num_products){
            // PROCESS THE BATCH INDEX
            forward_kernel_shared_memory(L1_smem, L2_smem, L3_smem, weights);
        }
        __syncthreads();
        
        // WRITE OUT TO GLOBAL 
        contiguous_copy(L3_smem, L3_shift, block_batch_elements_to_process * {{L3.rep_len}});         
    } 
}




// Generic kernel which calls a subkernel for each backward interaction
// column-major data layout
/*
* Backward pass kernel. Currently assumes that each tensor product
* has a unique set of weights. 
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
    return;
}

// UTILITY FUNCTIONS HERE

__device__ __forceinline__ void contiguous_copy(float* src_ptr, float* dst_ptr, int n){
    for(int i = 0; i < n; i+= blockDim.x){
        if (i + threadIdx.x < n){
            dst_ptr[i + threadIdx.x] = src_ptr[i + threadIdx.x]; 
        }
    }
} 

__device__ __forceinline__ void contiguous_set(float* ptr, float value, int n){
    for(int i = 0; i < n; i+= blockDim.x){
        if (i + threadIdx.x < n){
            ptr[i + threadIdx.x] = value; 
        }
    }
}


// All the subkernels must be appended to this file