{# Jinja2 Tempate #}
constexpr int FORWARD_THREADS_PER_WARP = {{forward_config.warp_size}};
constexpr int BACKWARD_THREADS_PER_WARP = {{backward_config.warp_size}};

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

{%- from 'macros.jinja' import declare_smem_arrays with context %}

// FORWARD DECLARATIONS
__device__ __forceinline__ void contiguous_copy (float*, float* , int); 
__device__ __forceinline__ void contiguous_set  (float*, float  , int);

// DOES ONE BATCH ELEMENT
__device__ __forceinline__ void forward_kernel_shared_memory(
    float* __restrict__ L1_smem, 
    float* __restrict__ L2_smem,
    float* __restrict__ smem_gemm_L3, 
    float* __restrict__ smem_gemm_weights,  
    float* __restrict__ L3_smem,
    float* __restrict__ weights 
    ){ 
    // This function expects a to be pointed at a location in shared memory 
    // It will perform a fixed number of batch elements per execution
    
    float L1_local_vec[{{L1.irrep_lengths | max}}];
    float L2_local_vec[{{L2.irrep_lengths | max}}];
    float L3_local_vec[{{L3.irrep_lengths | max}}];
    
    auto group = cooperative_groups::this_thread_block(); 

    contiguous_set(L3_smem, 0.0f, {{L3.rep_len}});
    
    // Loop Through Interactions
    {%- set num_interact = interactions | length %}
    {%- for interaction_index in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[interaction_index] %}
        {%- set weight_offset = weight_offsets[interaction_index] %}
    {    
        // Calcualate all shifts

        float* L1_smem_interaction_shift = L1_smem + {{L1.offsets[u]}}; 
        float* L2_smem_interaction_shift = L2_smem + {{L2.offsets[v]}};
        float* L3_smem_interaction_shift = L3_smem + {{L3.offsets[w]}};  
        float* weights_interaction_shift = weights + {{weight_offset}}; 
        
        int L1_mults = {{L1.mults[u]}};
        int L2_mults = {{L2.mults[v]}};
        int L3_mults = {{L3.mults[w]}};
        
        int L1_irrep_length = {{L1.irrep_lengths[u]}}; 
        int L2_irrep_length = {{L2.irrep_lengths[v]}};
        int L3_irrep_length = {{L3.irrep_lengths[w]}}; 

        int num_L1_L2_combinations = L1_mults * L2_mults; 

        // Assign Each Thread one multiplictiy interaction, BlockDim stride kernel  
        for (int L1_L2_block_start_index = 0; 
             L1_L2_block_start_index < num_L1_L2_combinations; 
             L1_L2_block_start_index += blockDim.x
            ){
            
            int num_L1_L2_blocks_to_do = min(blockDim.x, num_L1_L2_combinations - L1_L2_block_start_index);
            int n = num_L1_L2_blocks_to_do; 

            int L1_L2_thread_index = L1_L2_block_start_index + threadIdx.x; 
            int L1_mult_index = L1_L2_thread_index / L2_mults; 
            int L2_mult_index = L1_L2_thread_index % L2_mults; 

            const float* L1_smem_multiplicity_shift = L1_smem_interaction_shift + (L1_mult_index * L1_irrep_length);
            const float* L2_smem_multiplicity_shift = L2_smem_interaction_shift + (L2_mult_index * L2_irrep_length); 

            // READ N SETS OF WEIGHTS FROM GLOBAL MEMORY, 
            // HOPEFULLY L2 CACHE TO SMEM COLUMN MAJOR ORDER
            float* weights_mult1_mult2_shift = weights_interaction_shift + (L1_L2_block_start_index * {{L3.mults[w]}}); 
            cooperative_groups::memcpy_async(group, smem_gemm_weights, weights_mult1_mult2_shift, sizeof(float) * n * {{L3.mults[w]}});            


            // N threads perform a tensor product 
            if (threadIdx.x < n) {
                
                // CLEAR L3 REGISTERS
                #pragma unroll
                for(int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                        L3_local_vec[L3_irrep_index] = 0.0f;
                }

                // LOAD L1_SMEM INTO REGISTERS 
                #pragma unroll
                for (int L1_irrep_index = 0; L1_irrep_index < {{L1.irrep_lengths[u]}}; L1_irrep_index++){
                    L1_local_vec[L1_irrep_index] = L1_smem_multiplicity_shift[L1_irrep_index];  
                }

                // LOAD L2_SMEM INTO REGISTERS
                #pragma unroll 
                for(int L2_irrep_index = 0; L2_irrep_index < {{L2.irrep_lengths[v]}}; L2_irrep_index++){
                    L2_local_vec[L2_irrep_index] = L2_smem_multiplicity_shift[L2_irrep_index];
                }

                // PERFORM CG DECOMPOSITION CALCULATION 
                {%- for i in range(tensor.nnz) %}
                    {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                    L3_local_vec[{{coord3}}] += {{value}} * {{instructions[interaction_index].path_weight}} * L1_local_vec[{{coord1}}] * L2_local_vec[{{coord2}}];
                {%- endfor %}

                // WRITE TO SMEM_GEMM_L3 
                #pragma unroll 
                for(int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                    smem_gemm_L3[(threadIdx.x * {{L3.irrep_lengths[w]}}) + L3_irrep_index] = L3_local_vec[L3_irrep_index]; 
                }
            }

            // WAIT FOR WEIGHTS TO HAVE ARRIVED
            cooperative_groups::wait(group); 

            group.sync();

            // PERFORM MATMUL 
            int i = threadIdx.x;
            if(i < L3_mults){
                float accumulators[{{L3.irrep_lengths[w]}}]; 
                
                #pragma unroll 
                for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
                    accumulators[j]=0; 
                }

                for(int k = 0; k < n; k++){
                    float local_weight = smem_gemm_weights[(k  * L3_mults) + i];
                    #pragma unroll 
                    for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
                    //    C[i j]    +=    A[i k]    *     B[kj]
                    accumulators[j] += local_weight * smem_gemm_L3[(k * {{L3.irrep_lengths[w]}}) + j];
                    }
                } 

                #pragma unroll
                for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
                    L3_smem_interaction_shift[(i * {{L3.irrep_lengths[w]}}) + j] += accumulators[j];
                }   
            }

            group.sync();  
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
    float* L3_out,
    float* weights
    ) 
{   
    {%- set warps_per_block = divide(forward_config.num_threads, forward_config.warp_size) %}
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / FORWARD_THREADS_PER_WARP; 
    int lane_id = idx % FORWARD_THREADS_PER_WARP;
    int warp_loc = warp_id % {{warps_per_block}}; 

    {{ declare_smem_arrays( { 
        "common": [
            ("L1_smem", "float", L1.rep_len),
            ("L2_smem", "float", L2.rep_len), 
            ("L3_smem", "float", L3.rep_len),
            ("smem_gemm_L3", "float", smem_gemm_info.L3_scratch_elems),
            ("smem_gemm_weights", "float", smem_gemm_info.weight_scratch_elems), 
            ], 
        "per_warp": []}, "warp_loc", forward_config)}}    

    auto group = cooperative_groups::this_thread_block(); 
    
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier; 
    
    if (group.thread_rank() == 0){
        init(&barrier, group.size());
    }
    
    {%- for II in InstructionInfoList %}
    {
        float* instruction_weights = weights + {{II.weight_offset}};
        int instruction_num_weights = {{math.prod(II.weight_shape)}}; 

        

    }
    {% endfor %}    


    group.sync(); 
    // GRID STRIDE LOOP
    for(
        int batch_index = blockIdx.x; 
        batch_index < num_products; 
        batch_index += gridDim.x
        )
    {   
        // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
        float* L1_shift = L1_in  + (batch_index * {{L1.rep_len}}); 
        float* L2_shift = L2_in  + (batch_index * {{L2.rep_len}}); 
        float* L3_shift = L3_out + (batch_index * {{L3.rep_len}}); 
    
        // COPY FROM GLOBAL
        cuda::memcpy_async(group, L1_smem, L1_shift, sizeof(float) * {{L1.rep_len}}, barrier);
        cuda::memcpy_async(group, L2_smem, L2_shift, sizeof(float) * {{L2.rep_len}}, barrier);              

        barrier.arrive_and_wait();
        
        // PERFORM TP 
        forward_kernel_shared_memory(L1_smem, L2_smem, smem_gemm_L3, smem_gemm_weights, L3_smem, weights);

        group.sync();
        
        // WRITE TO GLOBAL
        cuda::memcpy_async(group, L3_shift, L3_smem, sizeof(float) * {{L3.rep_len}}, barrier);        
    } 
}

__device__ __forceinline__ void backward_kernel_shared_memory(
    float* __restrict__ L1_smem,
    float* __restrict__ L1_grad_smem,  
    float* __restrict__ L2_smem,
    float* __restrict__ L2_grad_smem, 
    float* __restrict__ L3_grad_smem,
    float* __restrict__ weights,     // global
    float* __restrict__ weights_grad // global 
    )
{
    float L1_local_vec[{{L1.irrep_lengths | max}}];
    float L1_grad_local_vec[{{L1.irrep_lengths | max}}];
    float L2_local_vec[{{L2.irrep_lengths | max}}];
    float L2_grad_local_vec[{{L2.irrep_lengths | max}}];
    float L3_grad_local_vec[{{L3.irrep_lengths | max}}];

    {% set num_scratch_reg = 1 %}
    float scratch1[{{num_scratch_reg}}];
    float scratch2[{{num_scratch_reg}}];


    auto group = cooperative_groups::this_thread_block();

    contiguous_set(L1_grad_smem, 0.0f, {{L1.rep_len}});
    contiguous_set(L2_grad_smem, 0.0f, {{L2.rep_len}}); 

    // Loop Through Interactions
    {%- set num_interact = interactions | length %}
    {%- for interaction_index in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[interaction_index] %}
        {%- set weight_offset = weight_offsets[interaction_index] %}
    {    
        float* L1_smem_interaction_shift = L1_smem + {{L1.offsets[u]}};
        float* L1_grad_smem_interaction_shift = L1_grad_smem + {{L1.offsets[u]}};
        float* L2_smem_interaction_shift = L2_smem + {{L2.offsets[v]}};
        float* L2_grad_smem_interaction_shift = L2_grad_smem + {{L2.offsets[v]}};
        float* L3_grad_smem_interaction_shift = L3_grad_smem + {{L3.offsets[w]}};  
        float* weights_interaction_shift = weights + {{weight_offset}}; 
        float* weights_grad_interaction_shift = weights_grad + {{weight_offset}}; 

        int L1_mults = {{L1.mults[u]}};
        int L2_mults = {{L2.mults[v]}};
        int L3_mults = {{L3.mults[w]}};
        
        int L1_irrep_length = {{L1.irrep_lengths[u]}}; 
        int L2_irrep_length = {{L2.irrep_lengths[v]}};
        int L3_irrep_length = {{L3.irrep_lengths[w]}}; 
        
        int num_weights = L1_mults * L2_mults * L3_mults;

        // assign every thread one weight LMAO, yes, this is very bad, correctness here we come
        for ( 
            int block_start_weight_index = 0;
            block_start_weight_index < num_weights;  
            block_start_weight_index += blockDim.x 
            )
            {
            
            int weight_index = block_start_weight_index + threadIdx.x; 

            int L1_mult_index = weight_index / (L2_mults * L3_mults);
            int remaining = weight_index % (L2_mults * L3_mults);
            int L2_mult_index = remaining / L3_mults;
            int L3_mult_index = remaining % L3_mults;  

            float* L1_smem_multiplicity_shift = L1_smem_interaction_shift + (L1_mult_index * L1_irrep_length); 
            float* L1_grad_smem_multiplicity_shift = L1_grad_smem_interaction_shift + (L1_mult_index * L1_irrep_length); 
            float* L2_smem_multiplicity_shift = L2_smem_interaction_shift + (L2_mult_index * L2_irrep_length); 
            float* L2_grad_smem_multiplicity_shift = L2_grad_smem_interaction_shift + (L2_mult_index * L2_irrep_length); 
            float* L3_grad_smem_multiplicity_shift = L3_grad_smem_interaction_shift + (L3_mult_index * L3_irrep_length);
            
            __syncwarp();

            if(weight_index < num_weights)
            {
                // LOAD DATA 

                // LOAD L1_SMEM INTO REGISTERS 
                #pragma unroll
                for (int L1_irrep_index = 0; L1_irrep_index < L1_irrep_length; L1_irrep_index++){
                    L1_local_vec[L1_irrep_index] = L1_smem_multiplicity_shift[L1_irrep_index];  
                }

                // LOAD L2_SMEM INTO REGISTERS
                #pragma unroll 
                for(int L2_irrep_index = 0; L2_irrep_index < L2_irrep_length; L2_irrep_index++){
                    L2_local_vec[L2_irrep_index] = L2_smem_multiplicity_shift[L2_irrep_index];
                }

                // LOAD L3_GRAD_SMEM INTO REGISTERS
                #pragma unroll 
                for(int L3_irrep_index = 0; L3_irrep_index < L3_irrep_length; L3_irrep_index++){
                    L3_grad_local_vec[L3_irrep_index] = L3_grad_smem_multiplicity_shift[L3_irrep_index];
                }
                
                // LOAD WEIGHT 
                float local_weight = weights_interaction_shift[weight_index]; 

                // ZERO ACUMULATORS 

                // ZERO L1_GRAD_LOCAL
                #pragma unroll
                for (int L1_irrep_index = 0; L1_irrep_index < L1_irrep_length; L1_irrep_index++){
                    L1_grad_local_vec[L1_irrep_index] = 0;  
                }

                // ZERO L2_GRAD_LOCAL
                #pragma unroll 
                for(int L2_irrep_index = 0; L2_irrep_index < L2_irrep_length; L2_irrep_index++){
                    L2_grad_local_vec[L2_irrep_index] = 0;
                }
                            
                // ZERO WEIGHT GRAD 
                float local_weight_grad = 0; 

                // BACKPROP THROUGH CG contraction
                {%- for i in range(tensor.nnz) %} 
                        {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                        scratch1[{{i % num_scratch_reg}}] = L3_grad_local_vec[{{coord3}}] * ({{value}}*{{instructions[interaction_index].path_weight}}); 
                        local_weight_grad += scratch1[{{i % num_scratch_reg}}] * L2_local_vec[{{coord2}}] * L1_local_vec[{{coord1}}];
                        scratch2[{{i % num_scratch_reg}}] = scratch1[{{i % num_scratch_reg}}] * local_weight;
                        L2_grad_local_vec[{{coord2}}] += scratch2[{{i % num_scratch_reg}}] * L1_local_vec[{{coord1}}];
                        L1_grad_local_vec[{{coord1}}] += scratch2[{{i % num_scratch_reg}}] * L2_local_vec[{{coord2}}];
                {%- endfor %}

                // STORE RESULTS 

                // BLOCKWIDE ATOMIC ADD L1_GRAD_LOCAL TO L1_GRAD_SMEM 
                #pragma unroll
                for (int L1_irrep_index = 0; L1_irrep_index < L1_irrep_length; L1_irrep_index++){
                    atomicAdd_block(&L1_grad_smem_multiplicity_shift[L1_irrep_index], L1_grad_local_vec[L1_irrep_index]);
                }
                

                // BLOCK-WIDE ATOMIC ADD L2_GRAD_LOCAL TO L2_GRAD_SMEM 
                #pragma unroll 
                for(int L2_irrep_index = 0; L2_irrep_index < L2_irrep_length; L2_irrep_index++){
                    atomicAdd_block(&L2_grad_smem_multiplicity_shift[L2_irrep_index], L2_grad_local_vec[L2_irrep_index]);
                }
                
                // DEVICE-WIDE ATOMIC ADD WEIGHT_GRAD 
                atomicAdd(&weights_grad_interaction_shift[weight_index], local_weight_grad); 
            }
            __syncwarp(); 
            }
        __syncthreads(); 
    }
    {%- endfor %}  
    

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
    {%- set warps_per_block = divide(forward_config.num_threads, forward_config.warp_size) %}
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / BACKWARD_THREADS_PER_WARP; 
    int lane_id = idx % BACKWARD_THREADS_PER_WARP;
    int warp_loc = warp_id % {{warps_per_block}}; 

    {{ declare_smem_arrays( { 
        "common": [
            ("L1_smem", "float", L1.rep_len),
            ("L1_grad_smem", "float", L1.rep_len), 
            ("L2_smem", "float", L2.rep_len),
            ("L2_grad_smem", "float", L2.rep_len), 
            ("L3_grad_smem", "float", L3.rep_len),
            ], 
        "per_warp": []}, "warp_loc", forward_config)}}    

    auto group = cooperative_groups::this_thread_block(); 
    
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier; 
    
    if (group.thread_rank() == 0){
        init(&barrier, group.size());
    }
    group.sync(); 
    contiguous_set(weights_grad, 0.0f, {{weight_numel}}); 

    group.sync(); 
    // GRID STRIDE LOOP
    for(
        int batch_index = blockIdx.x; 
        batch_index < num_products; 
        batch_index += gridDim.x
        )
    {   
        // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
        float* L1_shift = L1_in  + (batch_index * {{L1.rep_len}}); 
        float* L1_grad_shift = L1_grad + (batch_index * {{L1.rep_len}}); 
        float* L2_shift = L2_in  + (batch_index * {{L2.rep_len}}); 
        float* L2_grad_shift = L2_grad + (batch_index * {{L2.rep_len}}); 
        float* L3_grad_shift = L3_grad + (batch_index * {{L3.rep_len}}); 
    
        // COPY FROM GLOBAL
        cuda::memcpy_async(group, L1_smem, L1_shift, sizeof(float) * {{L1.rep_len}}, barrier);
        cuda::memcpy_async(group, L2_smem, L2_shift, sizeof(float) * {{L2.rep_len}}, barrier);           
        cuda::memcpy_async(group, L3_grad_smem, L3_grad_shift, sizeof(float) * {{L3.rep_len}}, barrier);    

        barrier.arrive_and_wait();
        
        // PERFORM TP 
        backward_kernel_shared_memory(L1_smem, L1_grad_smem, L2_smem, L2_grad_smem, L3_grad_smem, weights, weights_grad);

        group.sync();
        
        // WRITE TO GLOBAL
        cuda::memcpy_async(group, L1_grad_shift, L1_grad_smem, sizeof(float) * {{L1.rep_len}}, barrier); 
        cuda::memcpy_async(group, L2_grad_shift, L2_grad_smem, sizeof(float) * {{L2.rep_len}}, barrier);        
    } 
}

__device__ __forceinline__ void contiguous_copy(float* dst, float* src, int n){
    for(int i = 0; i < n; i+= blockDim.x){
        if (i + threadIdx.x < n){
            dst[i + threadIdx.x] = src[i + threadIdx.x]; 
        }
    }
}

// UTILITY FUNCTIONS HERE
__device__ __forceinline__ void contiguous_set(float* ptr, float value, int n){
    for(int i = 0; i < n; i+= blockDim.x){
        if (i + threadIdx.x < n){
            ptr[i + threadIdx.x] = value; 
        }
    }
}
