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
template<typename Tensor>
__device__ __forceinline__ void print_tensor_contents(Tensor t); 

// DOES ONE BATCH ELEMENT
template<typename Tile> 
__device__ __forceinline__ void forward_kernel_shared_memory(
    Tile tile,
    float* __restrict__ L1_smem_warp, 
    float* __restrict__ L2_smem_warp,
    float* __restrict__ gemm_L3_smem_warp, 
    float* __restrict__ gemm_weights_smem_warp,  
    float* __restrict__ L3_smem_warp,
    float* __restrict__ weights 
    ){ 
    // This function expects a to be pointed at a location in shared memory 
    // It will perform a fixed number of batch element`s per execution
    
    float L1_local_vec[{{L1.irrep_lengths | max}}];
    float L2_local_vec[{{L2.irrep_lengths | max}}];
    float L3_local_vec[{{L3.irrep_lengths | max}}];

    static_contiguous_set<float, Tile, {{L3.rep_len}}>(tile, L3_smem_warp, 0.0f);
    
    tile.sync(); 

    // Loop Through Interactions
    {%- set num_interact = interactions | length %}
    {%- for interaction_index in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[interaction_index] %}
        {%- set weight_offset = weight_offsets[interaction_index] %}
    {   
        // Get templated values out of jinja into C++  
        constexpr int L1_mults = {{L1.mults[u]}};
        constexpr int L2_mults = {{L2.mults[v]}};
        constexpr int L3_mults = {{L3.mults[w]}};
        
        constexpr int L1_irrep_length = {{L1.irrep_lengths[u]}}; 
        constexpr int L2_irrep_length = {{L2.irrep_lengths[v]}};
        constexpr int L3_irrep_length = {{L3.irrep_lengths[w]}}; 
        
        // Calcualate all shifts
        float* L1_smem_interaction_shift = L1_smem_warp + {{L1.offsets[u]}}; 
        float* L2_smem_interaction_shift = L2_smem_warp + {{L2.offsets[v]}};
        float* L3_smem_interaction_shift = L3_smem_warp + {{L3.offsets[w]}};  
        float* weights_interaction_shift = weights + {{weight_offset}}; 

        // CUTLASS STUFF 

        // Tensor Definitions
        // C = A * B 
        
        // C : L3       {irrep_length, L3_mults} column_major 
        // B : L1L2     {irrep_length, 32      } column_major
        // A : Weights  {32       , L3_mults} row_major  

        // M, N, K Definition
        constexpr int gemm_M = L3_irrep_length;
        constexpr int gemm_N = L3_mults; 
        constexpr int gemm_K = tile.size(); 
        
        // Create CuTe problem shape

        // auto shape_MNK = cute::make_shape(gemm_M, gemm_N, gemm_K); 


        // CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{});  

        // SMEM TENSORS 
        // Create Smem Layouts
        cute::Layout layout_A_smem = cute::make_layout(cute::make_shape(cute::Int<gemm_M>{}, cute::Int<gemm_K>{}), cute::LayoutLeft{}); // Column Major
        cute::Layout layout_B_smem = cute::make_layout(cute::make_shape(cute::Int<gemm_N>{}, cute::Int<gemm_K>{}), cute::LayoutLeft{}); // Column Major
        cute::Layout layout_C_smem = cute::make_layout(cute::make_shape(cute::Int<gemm_M>{}, cute::Int<gemm_N>{}), cute::LayoutLeft{}); // Column Major  
        // cute::Layout layout_output = cute::make_layout(cute::make_shape(cute::Int<gemm_M>{}, cute::Int<gemm_N>{}), cute::LayoutLeft{}); // Column Major  

        // Create Smem Tensors 
        cute::Tensor tensor_A_smem = cute::make_tensor(cute::make_smem_ptr(gemm_L3_smem_warp),         layout_A_smem);
        cute::Tensor tensor_B_smem = cute::make_tensor(cute::make_smem_ptr(gemm_weights_smem_warp),    layout_B_smem); 
        cute::Tensor tensor_C_smem = cute::make_tensor(cute::make_smem_ptr(L3_smem_interaction_shift), layout_C_smem); 
        // cute::Tensor tensor_output = cute::make_tensor(cute::make_smem_ptr(L3_smem_)) 

        // THREAD TENSORS
        // Create thread layouts
        // cute::Layout layout_A_threads = cute::make_layout(cute::make_shape(cute::Int< 1>{}, cute::Int<32>{}));
        // cute::Layout layout_B_threads = cute::make_layout(cute::make_shape(cute::Int< 1>{}, cute::Int<32>{}));
        cute::Layout layout_C_threads = cute::make_layout(cute::make_shape(cute::Int< 1>{}, cute::Int<tile.size()>{})); 

        // CUTE_STATIC_ASSERT_V(cute::size(layout_A_threads) == cute::size(layout_B_threads));        
        // CUTE_STATIC_ASSERT_V(cute::size(layout_A_threads) == cute::size(layout_C_threads)); 
        
   

         // ALL OF THIS IS RELATED TO TUTORIAL 1 
            
            // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles
            // cute::Tensor layout_A_thread_partitioning_smem_A = cute::local_partition(tensor_A_smem, layout_A_threads, tile.thread_rank());
            // cute::Tensor layout_B_thread_partitioning_smem_B = cute::local_partition(tensor_B_smem, layout_B_threads, tile.thread_rank()); 

            // Partition sA (M,K) by the rows of tC
            cute::Tensor layout_C_thread_partioning_smem_A = cute::local_partition(tensor_A_smem, layout_C_threads, tile.thread_rank(), cute::Step<cute::_1,cute::X>{});   // (THR_M,BLK_K)
            // Partition sB (N,K) by the cols of tC
            cute::Tensor layout_C_thread_partioning_smem_B = cute::local_partition(tensor_B_smem, layout_C_threads, tile.thread_rank(), cute::Step<cute::X,cute::_1>{});   // (THR_N,BLK_K)
            // Partition sC (M,N) by the tile of tC
            cute::Tensor layout_C_thread_partioning_smem_C = cute::local_partition(tensor_C_smem, layout_C_threads, tile.thread_rank(), cute::Step<cute::_1,cute::_1>{});   // (THR_M,THR_N)
            
            // Allocate the accumulators -- same shape/layout as the partitioned data
            cute::Tensor layout_C_thread_partitioning_smem_C_registers = cute::make_tensor_like(layout_C_thread_partioning_smem_C);
            
            // // // PREDICATING 
            // cute::Tensor tCpC = cute::make_tensor<bool>(cute::make_shape(cute::size<0>(layout_C_thread_partioning_smem_C), cute::size<1>(layout_C_thread_partioning_smem_C)),cute::make_stride(cute::Int<1>{}, cute::Int<0>{})); 

            // cute::Tensor cC = cute::make_identity_tensor(cute::make_shape(cute::size<0>(tensor_C_smem),cute::size<1>(tensor_C_smem))); 

            // cute::Tensor tCcC = cute::local_partition(cC, layout_C_threads, tile.thread_rank()); 

            // CUTE_UNROLL
            // for (int m = 0; m < cute::size<0>(tCpC); ++m){
            //     CUTE_UNROLL 
            //     for (int n = 0; n < cute::size<1>(tCpC); ++n){
            //            tCpC(m,n) = (cute::get<0>(cute::get<0>(tCcC(m,n))) < cute::size<0>(cC)) || (cute::get<0>(cute::get<1>(tCcC(m,n)) < cute::size<1>(cC))); 
            //     }             
            // }

           

            // END OF STUFF RELATED TO Tutorial 1

            // TUTORIAL 2 ################################################
            // cute::TiledMMA tiled_mma = cute::make_tiled_mma(
            //     cute::UniversalFMA<float, float, float>{},
            //     cute::Layout<cute::Shape<cute::_1,cute::_32>>{} // mnk? 
            //     ); 

            // cute::ThrMMA thread_mma = tiled_mma.get_slice(tile.thread_rank()); 
            
            // cute::Tensor layout_C_thread_partioning_smem_A = thread_mma.partition_A(tensor_A_smem); 
            // cute::Tensor layout_C_thread_partioning_smem_B = thread_mma.partition_B(tensor_B_smem); 
            // cute::Tensor layout_C_thread_partioning_smem_C = thread_mma.partition_C(tensor_C_smem); 

            // cute::Tensor layout_C_thread_partitioning_smem_C_registers = thread_mma.make_fragment_C(layout_C_thread_partioning_smem_C); 

            // CUTE_STATIC_ASSERT_V(cute::size<1>(layout_C_thread_partioning_smem_C) == cute::size<1>(layout_C_thread_partioning_smem_A));                // MMA_M
            // CUTE_STATIC_ASSERT_V(cute::size<2>(layout_C_thread_partioning_smem_C) == cute::size<1>(layout_C_thread_partioning_smem_B));                // MMA_N
            // CUTE_STATIC_ASSERT_V(cute::size<2>(layout_C_thread_partioning_smem_A) == cute::size<2>(layout_C_thread_partioning_smem_B));                // MMA_K
            

            // TUTORIAL 2
       
   
        constexpr int num_L1_L2_combinations = L1_mults * L2_mults; 

        // Assign Each Thread one multiplictiy interaction, tile.size() stride kernel  
        for (int L1_L2_warp_start_index = 0; 
             L1_L2_warp_start_index < num_L1_L2_combinations; 
             L1_L2_warp_start_index += tile.size()
            ){
            
            int num_L1_L2_blocks_to_do = min(tile.size(), num_L1_L2_combinations - L1_L2_warp_start_index);
            int n = num_L1_L2_blocks_to_do; 

            int L1_L2_thread_index = L1_L2_warp_start_index + tile.thread_rank(); 
            int L1_mult_index = L1_L2_thread_index / L2_mults; 
            int L2_mult_index = L1_L2_thread_index % L2_mults; 

            const float* L1_smem_multiplicity_shift = L1_smem_interaction_shift + (L1_mult_index * L1_irrep_length);
            const float* L2_smem_multiplicity_shift = L2_smem_interaction_shift + (L2_mult_index * L2_irrep_length); 

            // READ N SETS OF WEIGHTS FROM GLOBAL MEMORY, 
            // HOPEFULLY L2 CACHE TO SMEM COLUMN MAJOR ORDER
            float* weights_mult1_mult2_shift = weights_interaction_shift + (L1_L2_warp_start_index * {{L3.mults[w]}});           
            // dynamic_contiguous_copy<float, Tile>(tile, gemm_weights_smem_warp, weights_mult1_mult2_shift, n * L3_mults); 
            
            bool active_for_tensor_product = (tile.thread_rank() < n);

            #pragma unroll
            for(int L3_mult_index = 0; L3_mult_index < L3_mults; L3_mult_index++){
                tensor_B_smem(L3_mult_index, tile.thread_rank()) = active_for_tensor_product ? weights_mult1_mult2_shift[(tile.thread_rank() * L3_mults) + L3_mult_index] : 0.0f; 
            }

            // N threads perform a tensor product 
            if (active_for_tensor_product){
                
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
            }

            // WRITE TO SMEM_GEMM_L3 
            #pragma unroll 
            for(int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                gemm_L3_smem_warp[(tile.thread_rank() * {{L3.irrep_lengths[w]}}) + L3_irrep_index] = (active_for_tensor_product ? L3_local_vec[L3_irrep_index] : 0.0f); 
            }

           
            // 

            // zero registers
            

            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
                // cute::print("layout_A_threads: "); cute::print(layout_A_threads); cute::print("\n");
                // cute::print("layout_B_threads: "); cute::print(layout_B_threads); cute::print("\n");
                cute::print("layout_C_threads: "); cute::print(layout_C_threads); cute::print("\n");
            }
            #endif

            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
                cute::print("  sA : "); cute::print(tensor_A_smem); cute::print("\n");
                cute::print("  sB : "); cute::print(tensor_B_smem); cute::print("\n");
                cute::print("  sC : "); cute::print(tensor_C_smem); cute::print("\n");
                cute::print("\n");
            }
            #endif

            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
                
                cute::print("tCsA : "); cute::print(layout_C_thread_partioning_smem_A); cute::print("\n");
                cute::print("tCsB : "); cute::print(layout_C_thread_partioning_smem_B); cute::print("\n");
                cute::print("tCrC : "); cute::print(layout_C_thread_partioning_smem_C); cute::print("\n");
                cute::print("\n");
            }
            #endif
            
            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
                cute::print("Pre Multiplication State:\n");
                cute::print("sA :\n"); print_tensor_contents(tensor_A_smem); cute::print("\n");
                cute::print("sB :\n"); print_tensor_contents(tensor_B_smem); cute::print("\n");
                cute::print("sC :\n"); print_tensor_contents(tensor_C_smem); cute::print("\n");
                cute::print("\n");
            }
            #endif


            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
           
                // cute::print("tCsA :\n"); print_tensor_contents(layout_C_thread_partioning_smem_A); cute::print("\n");
                // cute::print("tCsB :\n"); print_tensor_contents(layout_C_thread_partioning_smem_B); cute::print("\n");
                // cute::print("tCrC :\n"); print_tensor_contents(layout_C_thread_partioning_smem_C); cute::print("\n");
                cute::print("\n");
            }
            #endif

            // auto K_TILE_MAX = cute::size<2>(layout_A_thread_partitioning_smem_A); this dosn't work. 

            // for(int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile){
            //     // Copy gmem to smem with tA|tB thread-partitioned tensors
            //     copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
            //     copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

            //     cute::cp_async_fence();   // Label the end of (potential) cp.async instructions
            //     cute::cp_async_wait<0>(); // Sync on all (potential) cp.async instructions
                
                tile.sync();              // Wait for all threads to write to smem
                
                cute::clear(layout_C_thread_partitioning_smem_C_registers); 
                // cute::gemm(tiled_mma, layout_C_thread_partioning_smem_A, layout_C_thread_partioning_smem_B, layout_C_thread_partitioning_smem_C_registers);
                cute::gemm(layout_C_thread_partioning_smem_A, layout_C_thread_partioning_smem_B, layout_C_thread_partitioning_smem_C_registers); // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
     
                // if(tile.thread_rank() == 0){
                    // CUTE_UNROLL
                    // for (int k = 0; k < cute::size<1>(layout_C_thread_partioning_smem_A); ++k) {
                    //     CUTE_UNROLL
                    //     for (int m = 0; m < cute::size<0>(layout_C_thread_partioning_smem_C); ++m) {
                    //     CUTE_UNROLL
                    //     for (int n = 0; n < cute::size<1>(layout_C_thread_partioning_smem_C); ++n) {
                    //         layout_C_thread_partitioning_smem_C_registers(m,n) += layout_C_thread_partioning_smem_A(m,k) * layout_C_thread_partioning_smem_B(n,k);
                    //     }
                    //     }
                    // } 
                // // }
                
                tile.sync();
            // }

            // cute::copy(layout_C_thread_partitioning_smem_C_registers, layout_C_thread_partioning_smem_C); 
            // cute::copy_if(tCpC, layout_C_thread_partitioning_smem_C_registers, layout_C_thread_partioning_smem_C); 
            
            int i = tile.thread_rank();
            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
                L3_smem_interaction_shift[(i * {{L3.irrep_lengths[w]}}) + j] += layout_C_thread_partitioning_smem_C_registers(j,i);
            }
        

            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
                cute::print("Post Multiplciation State:\n");
                cute::print("sA :\n"); print_tensor_contents(tensor_A_smem); cute::print("\n");
                cute::print("sB :\n"); print_tensor_contents(tensor_B_smem); cute::print("\n");
                cute::print("sC :\n"); print_tensor_contents(tensor_C_smem); cute::print("\n");
                cute::print("\n");
            }
            #endif

            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
           
                // cute::print("tCsA :\n"); print_tensor_contents(layout_C_thread_partioning_smem_A); cute::print("\n");
                // cute::print("tCsB :\n"); print_tensor_contents(layout_C_thread_partioning_smem_B); cute::print("\n");
                // cute::print("tCrC :\n"); print_tensor_contents(layout_C_thread_partioning_smem_C); cute::print("\n");
                cute::print("\n");
            }
            #endif

            // // PERFORM MATMUL 
            // int i = tile.thread_rank();
            // if(i < L3_mults){
            //     float accumulators[{{L3.irrep_lengths[w]}}]; 
                
            //     #pragma unroll 
            //     for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
            //         accumulators[j]=0; 
            //     }

            //     for(int k = 0; k < n; k++){
            //         float local_weight = gemm_weights_smem_warp[(k  * L3_mults) + i];
            //         #pragma unroll 
            //         for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
            //         //    C[i j]    +=    A[i k]    *     B[kj]
            //         accumulators[j] += local_weight * gemm_L3_smem_warp[(k * {{L3.irrep_lengths[w]}}) + j];
            //         }
            //     } 

            //     #pragma unroll
            //     for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++){
            //         L3_smem_interaction_shift[(i * {{L3.irrep_lengths[w]}}) + j] += accumulators[j];
            //     }   
            // }

            tile.sync();  
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
    constexpr int warps_per_block = {{warps_per_block}}; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP; 
    // int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % {{warps_per_block}}; 

    {{ declare_smem_arrays( { 
        "common": [
            ], 
        "per_warp": [
            ("L1_smem_warp", "float", L1.rep_len),
            ("L2_smem_warp", "float", L2.rep_len), 
            ("L3_smem_warp", "float", L3.rep_len),
            ("gemm_L3_smem_warp", "float", forward_smem_gemm_info.L3_scratch_elems),
            ("gemm_weights_smem_warp", "float", forward_smem_gemm_info.weight_scratch_elems), 
        ]
        }, "warp_loc", forward_config)}}    

    Warp_Tile warp_tile = cg::tiled_partition<THREADS_PER_WARP>(cg::this_thread_block());

    // GRID STRIDE LOOP
    for(
        int batch_index = (blockIdx.x * warps_per_block) + warp_loc; 
        batch_index < num_products; 
        batch_index += (gridDim.x * warps_per_block) 
        )
    {   
        // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
        float* L1_global_shift_warp = L1_in  + (batch_index * {{L1.rep_len}}); 
        float* L2_global_shift_warp = L2_in  + (batch_index * {{L2.rep_len}}); 
        float* L3_global_shift_warp = L3_out + (batch_index * {{L3.rep_len}}); 

        // COPY FROM GLOBAL
        static_contiguous_copy<float, Warp_Tile, {{L1.rep_len}}> (warp_tile, L1_smem_warp, L1_global_shift_warp);
        static_contiguous_copy<float, Warp_Tile, {{L2.rep_len}}> (warp_tile, L2_smem_warp, L2_global_shift_warp);

        warp_tile.sync(); 

        // PERFORM TP 
        forward_kernel_shared_memory<Warp_Tile>(
            warp_tile, 
            L1_smem_warp, 
            L2_smem_warp, 
            gemm_L3_smem_warp,
            gemm_weights_smem_warp, 
            L3_smem_warp, 
            weights
            );

        warp_tile.sync();
        
        // WRITE TO GLOBAL
        static_contiguous_copy<float, Warp_Tile, {{L3.rep_len}}> (warp_tile, L3_global_shift_warp, L3_smem_warp);
    } 
}

template<typename Tile>
__device__ __forceinline__ void backward_kernel_shared_memory(
    Tile tile, 
    float* __restrict__ L1_shared_shift_warp,
    float* __restrict__ L1_grad_shared_shift_warp,  
    float* __restrict__ L2_shared_shift_warp,
    float* __restrict__ L2_grad_shared_shift_warp, 
    float* __restrict__ L3_grad_shared_shift_warp,
    float* __restrict__ weights,     // global
    float* __restrict__ weights_grad, // global 
    float* __restrict__ gemm_L1L2_multipurpose_shared_warp, 
    float* __restrict__ gemm_weights_multipurpose_shared_warp
    )
{
    float L1_local_vec[{{L1.irrep_lengths | max}}];
    float L1_grad_local_vec[{{L1.irrep_lengths | max}}];
    float L2_local_vec[{{L2.irrep_lengths | max}}];
    float L2_grad_local_vec[{{L2.irrep_lengths | max}}];
    float L1L2_multipurpose_local_vec[{{L3.irrep_lengths | max}}];

    {% set num_scratch_reg = 1 %}
    float scratch[{{num_scratch_reg}}];

    cg::thread_block block = cg::this_thread_block();

    dynamic_contiguous_set<float,Tile>(tile, L1_grad_shared_shift_warp, 0.0f, {{L1.rep_len}});
    dynamic_contiguous_set<float,Tile>(tile, L2_grad_shared_shift_warp, 0.0f, {{L2.rep_len}}); 

    // Loop Through Interactions
    {%- set num_interact = interactions | length %}
    {%- for interaction_index in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[interaction_index] %}
        {%- set weight_offset = weight_offsets[interaction_index] %}
    {    
        float* L1_shared_shift_interaction = L1_shared_shift_warp + {{L1.offsets[u]}};
        float* L1_grad_shared_shift_interaction = L1_grad_shared_shift_warp + {{L1.offsets[u]}};
        float* L2_shared_shift_interaction = L2_shared_shift_warp + {{L2.offsets[v]}};
        float* L2_grad_shared_shift_interaction = L2_grad_shared_shift_warp + {{L2.offsets[v]}};
        float* L3_grad_shared_shift_interaction = L3_grad_shared_shift_warp + {{L3.offsets[w]}};  
        float* weights_global_shift_interaction = weights + {{weight_offset}}; 
        float* weights_grad_global_shift_interaction = weights_grad + {{weight_offset}}; 

        constexpr int L1_mults = {{L1.mults[u]}};
        constexpr int L2_mults = {{L2.mults[v]}};
        constexpr int L3_mults = {{L3.mults[w]}};
        
        constexpr int L1_irrep_length = {{L1.irrep_lengths[u]}}; 
        constexpr int L2_irrep_length = {{L2.irrep_lengths[v]}};
        constexpr int L3_irrep_length = {{L3.irrep_lengths[w]}}; 

        // Defintion of GEMM dimensions
        // I am not using M,N,K becuase they will be used in different orientations
        
        constexpr int gemm_I = L3_irrep_length; 
        constexpr int gemm_M = L3_mults; 
        constexpr int gemm_T = tile.size();  

        // FIRST MATMUL 
        // weights_grad (TILE_SIZE, L3_MULTS) = L1L2.transpose(TILE_SIZE,IRREP_DIM) * L3_grad(IRREP_DIM,L3_MULTS)
        // weights_grad (T, M) = L1L2.transpose(T,I) * L3_grad(I,M)
        // (T,M) = (T,I) x (I,M)
        // A = (T,I) : L1L2.transpose
        // B = (M,I) : L3_grad (flipped for cutlass conventions)
        // C = (T,M) : weights_grad

        // SECOND MATMUL 
        // L1L2_grad (irreps, tile.size) = L3_grad (irreps, L3_mults) x weights.transpose (L3_mults, tile.size) 
        // L1L2_grad (I, T) = L3_grad (I,M) x weights.transpose (M,T)
        // (I,T) = (I,M) x (M, T)
        // A = (I,M) : L3_grad
        // B = (T,M) : weights.transpose (flipped for cutlass conventions)
        // C = (I,T) : L1L2_grad
        
        // Layouts
        // Gemm 1
        cute::Layout layout_L1L2_smem         = cute::make_layout(cute::make_shape(cute::Int<gemm_T>{}, cute::Int<gemm_I>{}), cute::LayoutRight{}); // Row    Major (irrep_dim x tile.size)
        cute::Layout layout_L3_grad_smem_1    = cute::make_layout(cute::make_shape(cute::Int<gemm_M>{}, cute::Int<gemm_I>{}), cute::LayoutRight{}); // Row    Major (irrep_dim x L3_mults)
        cute::Layout layout_weights_grad_smem = cute::make_layout(cute::make_shape(cute::Int<gemm_T>{}, cute::Int<gemm_M>{}), cute::LayoutRight{}); // Row    Major (L3_mults  x tiles.size)
        // Gemm 2 
        cute::Layout layout_L3_grad_smem_2    = cute::make_layout(cute::make_shape(cute::Int<gemm_I>{}, cute::Int<gemm_M>{}), cute::LayoutLeft{} ); // Column Major (irrep_dim x L3_mults)
        cute::Layout layout_weights_smem      = cute::make_layout(cute::make_shape(cute::Int<gemm_T>{}, cute::Int<gemm_M>{}), cute::LayoutLeft{} ); // Column Major (tile.size x L3_mults)
        cute::Layout layout_L1L2_grad_smem    = cute::make_layout(cute::make_shape(cute::Int<gemm_I>{}, cute::Int<gemm_T>{}), cute::LayoutLeft{} ); // Column Major (irrep_dim x tile.size) 
        
        // Smem Tensors
        // Gemm 1
        cute::Tensor tensor_L1L2_smem         = cute::make_tensor(cute::make_smem_ptr(gemm_L1L2_multipurpose_shared_warp),    layout_L1L2_smem);
        cute::Tensor tensor_L3_grad_smem_1    = cute::make_tensor(cute::make_smem_ptr(L3_grad_shared_shift_interaction),      layout_L3_grad_smem_1);
        cute::Tensor tensor_weights_grad_smem = cute::make_tensor(cute::make_smem_ptr(gemm_weights_multipurpose_shared_warp), layout_weights_grad_smem);
        // Gemm 2 
        cute::Tensor tensor_L3_grad_smem_2    = cute::make_tensor(cute::make_smem_ptr(L3_grad_shared_shift_interaction),      layout_L3_grad_smem_2);
        cute::Tensor tensor_weights_smem      = cute::make_tensor(cute::make_smem_ptr(gemm_weights_multipurpose_shared_warp), layout_weights_smem);
        cute::Tensor tensor_L1L2_grad_smem    = cute::make_tensor(cute::make_smem_ptr(gemm_L1L2_multipurpose_shared_warp),    layout_L1L2_grad_smem); 
        
        // Thread Layout 
        cute::Layout layout_gemm_1 = cute::make_layout(cute::make_shape(cute::Int<gemm_T>{}, cute::Int<1>{})); 
        cute::Layout layout_gemm_2 = cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<gemm_T>{}));

        // Tensor Partitions
        // Gemm 1 
        cute::Tensor tensor_L1L2_thread         = cute::local_partition(tensor_L1L2_smem,         layout_gemm_1, tile.thread_rank(), cute::Step<cute::_1,cute::X> {}); 
        cute::Tensor tensor_L3_grad_thread_1    = cute::local_partition(tensor_L3_grad_smem_1,    layout_gemm_1, tile.thread_rank(), cute::Step<cute::X ,cute::_1>{});
        cute::Tensor tensor_weights_grad_thread = cute::local_partition(tensor_weights_grad_smem, layout_gemm_1, tile.thread_rank(), cute::Step<cute::_1,cute::_1>{});
        // Gemm 2
        cute::Tensor tensor_L3_grad_thread_2    = cute::local_partition(tensor_L3_grad_smem_2,    layout_gemm_2, tile.thread_rank(), cute::Step<cute::_1,cute::X>{});
        cute::Tensor tensor_weights_thread      = cute::local_partition(tensor_weights_smem,      layout_gemm_2, tile.thread_rank(), cute::Step<cute::X ,cute::_1>{});
        cute::Tensor tensor_L1L2_grad_thread    = cute::local_partition(tensor_L1L2_grad_smem,    layout_gemm_2, tile.thread_rank(), cute::Step<cute::_1,cute::_1>{});

        // Owned Tensors
        // Gemm 1
        cute::Tensor tensor_weights_grad_fragment = cute::make_tensor_like(tensor_weights_grad_thread); 
        // Gemm 2 
        cute::Tensor tensor_L1L2_grad_fragment = cute::make_tensor_like(tensor_L1L2_grad_thread); 

        // cute::Layout layout_L3_grad    = cute::make_layout(cute::make_shape(cute::Int<gemm_I>{}, cute::Int<gemm_M>{}), cute::LayoutLeft{});  // Column Major (irrep_dim x L3_mults)        
        // cute::Layout second_gemm_layout = cute::make_layout(cute::make_shape(cute::Int< 1>{}, cute::Int< tile.size()>));
        // cute::Layout tensor_L3_grad_thread_   = cute::local_partition(tensor_  , second_gemm_layout, tile.thread_rank(), cute::Step<cute::_1,cute::X> {});        
        
        constexpr int num_L1_L2_combinations = L1_mults * L2_mults; 

        // Assign Each Thread one multiplictiy interaction, tile.size() stride kernel  
        for (int L1_L2_warp_start_index = 0; 
             L1_L2_warp_start_index < num_L1_L2_combinations; 
             L1_L2_warp_start_index += tile.size()
            ){
            
            int num_L1_L2_blocks_to_do = min(tile.size(), num_L1_L2_combinations - L1_L2_warp_start_index);
            int n = num_L1_L2_blocks_to_do; 

            int L1_L2_thread_index = L1_L2_warp_start_index + tile.thread_rank(); 

            int L1_mult_index = L1_L2_thread_index / L2_mults; 
            int L2_mult_index = L1_L2_thread_index % L2_mults; 

            float* L1_smem_multiplicity_shift = L1_shared_shift_interaction + (L1_mult_index * L1_irrep_length); 
            float* L1_grad_smem_multiplicity_shift = L1_grad_shared_shift_interaction + (L1_mult_index * L1_irrep_length); 
            float* L2_smem_multiplicity_shift = L2_shared_shift_interaction + (L2_mult_index * L2_irrep_length); 
            float* L2_grad_smem_multiplicity_shift = L2_grad_shared_shift_interaction + (L2_mult_index * L2_irrep_length); 

            tile.sync();

            bool active_for_tensor_product = (tile.thread_rank() < n); 

            /*

            // CREATE L1L2 INTERMEDIATES (FORWARD)
            // THIS IS REQUIRED TO DO THE WEIGHT GRADIENT
            // N threads perform a tensor product 
            if (active_for_tensor_product) {
                
                // CLEAR L3 REGISTERS
                #pragma unroll
                for(int L3_irrep_index = 0; L3_irrep_index < L3_irrep_length; L3_irrep_index++){
                        L1L2_multipurpose_local_vec[L3_irrep_index] = 0.0f;
                }

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

                // PERFORM CG DECOMPOSITION CALCULATION 
                {%- for i in range(tensor.nnz) %}
                    {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                    L1L2_multipurpose_local_vec[{{coord3}}] += {{value}} * {{instructions[interaction_index].path_weight}} * L1_local_vec[{{coord1}}] * L2_local_vec[{{coord2}}];
                {%- endfor %}   
            }

            // WRITE TO GEMM_L1L2
 
            #pragma unroll 
            for(int L3_irrep_index = 0; L3_irrep_index < {{L3.irrep_lengths[w]}}; L3_irrep_index++){
                gemm_L1L2_multipurpose_shared_warp[(tile.thread_rank() * {{L3.irrep_lengths[w]}}) + L3_irrep_index] = active_for_tensor_product ? L1L2_multipurpose_local_vec[L3_irrep_index] : 0.0f; 
            }

            // GEMM 1 
            cute::clear(tensor_weights_grad_fragment); 
            cute::gemm(tensor_L1L2_thread, tensor_L3_grad_thread_1, tensor_weights_grad_fragment);

            tile.sync(); 

            // WRITE Weight Grad update to global
            // This is currently doing strided writes, I should write to shared first...
            for(int weight_L3_copy_index = 0; weight_L3_copy_index < L3_mults; weight_L3_copy_index++){
                if (active_for_tensor_product){
                    int weight_index = (L1_L2_thread_index * L3_mults) + weight_L3_copy_index; 
                    float local_weight_grad = tensor_weights_grad_fragment(0, weight_L3_copy_index);
                    // cute::print("weight_index : "); cute::print(weight_index); cute::print("\n"); 
                    // cute::print("local_weight_grad : "); cute::print(local_weight_grad); cute::print("\n"); 
                    // DEVICE-WIDE ATOMIC ADD WEIGHT_GRAD 
                    atomicAdd(&weights_grad_global_shift_interaction[weight_index], local_weight_grad); 
                }
            }

            #if 0
            tile.sync();
            if ((blockIdx.x == 0) && (threadIdx.x == 0)){
                cute::print("Post Multiplciation State:\n");
            }
            tile.sync();
            for(int thread_to_print = 0; thread_to_print < tile.size(); thread_to_print +=1){
                if((blockIdx.x == 0) && (threadIdx.x < tile.size()) && (tile.thread_rank() == thread_to_print)) {
                    cute::print("Thread : "); cute::print( tile.thread_rank()); cute::print("\n");
                    cute::print("weights fragment :\n"); print_tensor_contents(tensor_weights_grad_fragment); cute::print("\n");
                    cute::print("\n");
                }
                tile.sync(); 
            }
            #endif

            */
            
            tile.sync(); 
            // NOW WE ARE DOING THE L1L2 GRADIENTS
            // This is done by L1L2_grad = L3_grad x weights.transpose 
            // 

            // copy weights into tensor
            for (int L1L2_copy_index = 0; L1L2_copy_index < n; L1L2_copy_index ++){
                if(tile.thread_rank() < L3_mults){
                    int weight_index = ((L1_L2_warp_start_index + L1L2_copy_index) * L3_mults) + tile.thread_rank();  
                    tensor_weights_smem(L1L2_copy_index, tile.thread_rank()) = weights_global_shift_interaction[weight_index];
                }
            }

            // do nothing for L3 grad, because it should already be in smem? 

            // GEMM 2 
            cute::clear(tensor_L1L2_grad_fragment); 
            cute::gemm(tensor_L3_grad_thread_2, tensor_weights_thread, tensor_L1L2_grad_fragment);


            // load data from L1L2 smem into regiesters 

            #if DEBUG_PRINTING
            if(cute::thread0() && (n != tile.size())) {
                cute::print("Post Multiplciation State:\n");
                cute::print("sA :\n"); print_tensor_contents(tensor_A_smem); cute::print("\n");
                cute::print("sB :\n"); print_tensor_contents(tensor_B_smem); cute::print("\n");
                cute::print("sC :\n"); print_tensor_contents(tensor_C_smem); cute::print("\n");
                cute::print("\n");
            }
            #endif  

          
            #if 0
            tile.sync();
            if(cute::thread0()) {
                // cute::print("tCsA :\n"); print_tensor_contents(layout_C_thread_partioning_smem_A); cute::print("\n");
                // cute::print("tCsB :\n"); print_tensor_contents(layout_C_thread_partioning_smem_B); cute::print("\n");
                cute::print("tensor_L1L2_grad_thread layout : ");cute::print(tensor_L1L2_grad_thread);cute::print("\n");
                cute::print("tensor_L1L2_grad_thread :\n"); print_tensor_contents(tensor_L1L2_grad_thread); cute::print("\n");
                cute::print("tensor_L1L2_grad_fragment :\n"); print_tensor_contents(tensor_L1L2_grad_fragment); cute::print("\n");
                cute::print("\n");
            }
            #endif
            tile.sync();

            if(active_for_tensor_product)
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
                    L1L2_multipurpose_local_vec[L3_irrep_index] = tensor_L1L2_grad_fragment(L3_irrep_index, tile.thread_rank());
                }
                
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


                // BACKPROP THROUGH CG contraction
                {%- for i in range(tensor.nnz) %} 
                        {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                        scratch[{{i % num_scratch_reg}}] = L1L2_multipurpose_local_vec[{{coord3}}] * ({{value}}*{{instructions[interaction_index].path_weight}}); 
                        L2_grad_local_vec[{{coord2}}] += scratch[{{i % num_scratch_reg}}] * L1_local_vec[{{coord1}}];
                        L1_grad_local_vec[{{coord1}}] += scratch[{{i % num_scratch_reg}}] * L2_local_vec[{{coord2}}];
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
            }
            tile.sync(); 
            }
        block.sync(); 
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
    {%- set warps_per_block = divide(backward_config.num_threads, backward_config.warp_size) %}
    constexpr int warps_per_block = {{warps_per_block}}; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP; 
    // int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % warps_per_block; 

    {{ declare_smem_arrays( { 
        "common": [
            ], 
        "per_warp": [
            ("L1_shared_shift_warp", "float", L1.rep_len),
            ("L1_grad_shared_shift_warp", "float", L1.rep_len), 
            ("L2_shared_shift_warp", "float", L2.rep_len),
            ("L2_grad_shared_shift_warp", "float", L2.rep_len), 
            ("L3_grad_shared_shift_warp", "float", L3.rep_len),
            ("gemm_L1L2_smem_warp", "float", backward_smem_gemm_info.L1L2_scratch_elems),
            ("gemm_weights_smem_warp", "float", backward_smem_gemm_info.weight_scratch_elems),             
        ]
        }, "warp_loc", backward_config)}}    

    Warp_Tile warp_tile = cg::tiled_partition<THREADS_PER_WARP>(cg::this_thread_block());
    
    // GRID STRIDE LOOP
    for(
        int batch_index = (blockIdx.x * warps_per_block) + warp_loc; 
        batch_index < num_products; 
        batch_index += (gridDim.x * warps_per_block)
        )
    {   
        // THIS FOR LOOP TAKES INTO ACCOUNT THAT EACH WARP IS DOING A DIFFERENT BATCH ELEMENT 

        // CALCULATE PTRS TO THE ACTIVE BATCH ELEMENT REGION
        float* L1_global_shift_warp = L1_in  + (batch_index * {{L1.rep_len}}); 
        float* L1_grad_global_shift_warp = L1_grad + (batch_index * {{L1.rep_len}}); 
        float* L2_global_shift_warp = L2_in  + (batch_index * {{L2.rep_len}}); 
        float* L2_grad_global_shift_warp = L2_grad + (batch_index * {{L2.rep_len}}); 
        float* L3_grad_global_shift_warp = L3_grad + (batch_index * {{L3.rep_len}}); 
        
        // COPY FROM GLOBAL
        static_contiguous_copy<float, Warp_Tile, {{L1.rep_len}}>(warp_tile, L1_shared_shift_warp, L1_global_shift_warp); 
        static_contiguous_copy<float, Warp_Tile, {{L2.rep_len}}>(warp_tile, L2_shared_shift_warp, L2_global_shift_warp);
        static_contiguous_copy<float, Warp_Tile, {{L3.rep_len}}>(warp_tile, L3_grad_shared_shift_warp, L3_grad_global_shift_warp); 

        warp_tile.sync(); 

        // PERFORM TP 
        backward_kernel_shared_memory<Warp_Tile>(
            warp_tile, 
            L1_shared_shift_warp, 
            L1_grad_shared_shift_warp,  
            L2_shared_shift_warp, 
            L2_grad_shared_shift_warp, 
            L3_grad_shared_shift_warp, 
            weights, 
            weights_grad,
            gemm_L1L2_smem_warp, 
            gemm_weights_smem_warp
            );

        warp_tile.sync();
        
        // WRITE TO GLOBAL
        static_contiguous_copy<float, Warp_Tile, {{L1.rep_len}}>(warp_tile, L1_grad_global_shift_warp, L1_grad_shared_shift_warp);
        static_contiguous_copy<float, Warp_Tile, {{L2.rep_len}}>(warp_tile, L2_grad_global_shift_warp, L2_grad_shared_shift_warp);
    } 
}

// UTILITY FUNCTIONS HERE

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
