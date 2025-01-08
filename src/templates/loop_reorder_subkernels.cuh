{# Jinja2 Tempate #}

template<typename Tile>
__device__ __forceinline__ void forward_kernel_shared_memory_instruction_{{kernel_ID}}(
    Tile tile,
    float* __restrict__ L1_smem_interaction_shift, 
    float* __restrict__ L2_smem_interaction_shift,
    float* __restrict__ gemm_L3_smem_warp,
    float* __restrict__ gemm_weights_smem_warp,  
    float* __restrict__ L3_smem_interaction_shift,
    float* __restrict__ weights_smem_interaction_shift 
    ){ 
    // This function expects a to be pointed at a location in shared memory 
    // It will perform a fixed number of batch element`s per execution
    
    constexpr int L1_mults = {{II.in1_multiplicity}};
    constexpr int L2_mults = {{II.in2_multiplicity}};
    constexpr int L3_mults = {{II.out_multiplicity}};

    constexpr int L1_irrep_length = {{II.in1_irrep_length}}; 
    constexpr int L2_irrep_length = {{II.in2_irrep_length}};
    constexpr int L3_irrep_length = {{II.out_irrep_length}}; 

    constexpr int L1_size_instruction = L1_mults * L1_irrep_length;
    constexpr int L2_size_instruction = L2_mults * L2_irrep_length;
    constexpr int L3_size_instruction = L3_mults * L3_irrep_length;   

    float L1_local_vec[L1_irrep_length];
    float L2_local_vec[L2_irrep_length];
    float L3_local_vec[L3_irrep_length];
    
    tile.sync(); 

    
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
            float* weights_mult1_mult2_shift = weights_smem_interaction_shift + (L1_L2_warp_start_index * L3_mults);           
            dynamic_contiguous_copy<float, Tile>(tile, gemm_weights_smem_warp, weights_mult1_mult2_shift, n * L3_mults); 
            
            bool active_for_tensor_product = (tile.thread_rank() < n);
            
            // #pragma unroll
            // for(int L3_mult_index = 0; L3_mult_index < L3_mults; L3_mult_index++){
            //     tensor_B_smem(L3_mult_index, tile.thread_rank()) = active_for_tensor_product ? weights_mult1_mult2_shift[(tile.thread_rank() * L3_mults) + L3_mult_index] : 0.0f; 
            // }

            // N threads perform a tensor product 
            if (active_for_tensor_product){
                
                // CLEAR L3 REGISTERS
                #pragma unroll
                for(int L3_irrep_index = 0; L3_irrep_index < L3_irrep_length; L3_irrep_index++){
                        L3_local_vec[L3_irrep_index] = 0.0f;
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
                {%- for i in range(II.tensor.nnz) %}
                    {%- set coord1, coord2, coord3, value = II.tensor.tuples[i] %}
                    L3_local_vec[{{coord3}}] += {{value}} * {{II.path_weight}} * L1_local_vec[{{coord1}}] * L2_local_vec[{{coord2}}];
                {%- endfor %}   
            }

            // WRITE TO SMEM_GEMM_L3 
            #pragma unroll 
            for(int L3_irrep_index = 0; L3_irrep_index < L3_irrep_length; L3_irrep_index++){
                gemm_L3_smem_warp[(tile.thread_rank() * L3_irrep_length) + L3_irrep_index] = (active_for_tensor_product ? L3_local_vec[L3_irrep_index] : 0.0f); 
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
            for(int j = 0; j < L3_irrep_length; j++){
                L3_smem_interaction_shift[(i * L3_irrep_length) + j] += (i < L3_mults) ? layout_C_thread_partitioning_smem_C_registers(j,i) : 0.0f;
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
            //     float accumulators[L3_irrep_length]; 
                
            //     #pragma unroll 
            //     for(int j = 0; j < L3_irrep_length; j++){
            //         accumulators[j]=0; 
            //     }

            //     for(int k = 0; k < n; k++){
            //         float local_weight = gemm_weights_smem_warp[(k  * L3_mults) + i];
            //         #pragma unroll 
            //         for(int j = 0; j < L3_irrep_length}}; j++){
            //         //    C[i j]    +=    A[i k]    *     B[kj]
            //         accumulators[j] += local_weight * gemm_L3_smem_warp[(k * L3_irrep_length) + j];
            //         }
            //     } 

            //     #pragma unroll
            //     for(int j = 0; j < L3_irrep_length; j++){
            //         L3_smem_interaction_shift[(i * L3_irrep_length) + j] += accumulators[j];
            //     }   
            // }

            tile.sync();  
        }

}


template<typename Tile>
__device__ __forceinline__ void backward_kernel_shared_memory_instruction_{{kernel_ID}}(
    Tile tile, 
    float* __restrict__ L1_shared_shift_interaction,
    float* __restrict__ L1_grad_shared_shift_interaction,  
    float* __restrict__ L2_shared_shift_interaction,
    float* __restrict__ L2_grad_shared_shift_interaction, 
    float* __restrict__ L3_grad_shared_shift_interaction,
    float* __restrict__ weights_global_shift_interaction,     // global
    float* __restrict__ weights_grad_global_shift_interaction, // global 
    float* __restrict__ gemm_L1L2_multipurpose_shared_warp, 
    float* __restrict__ gemm_weights_multipurpose_shared_warp
    )
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

    float L1_local_vec[L1_irrep_length];
    float L1_grad_local_vec[L1_irrep_length];
    float L2_local_vec[L2_irrep_length];
    float L2_grad_local_vec[L2_irrep_length];
    float L1L2_multipurpose_local_vec[L3_irrep_length];

    {% set num_scratch_reg = 1 %}
    float scratch[{{num_scratch_reg}}];



    // dynamic_contiguous_set<float,Tile>(tile, L1_grad_shared_shift_interaction, 0.0f, L1_size_instruction);
    // dynamic_contiguous_set<float,Tile>(tile, L2_grad_shared_shift_interaction, 0.0f, L2_size_instruction); 

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
                {%- for i in range(II.tensor.nnz) %}
                    {%- set coord1, coord2, coord3, value = II.tensor.tuples[i] %}
                    L1L2_multipurpose_local_vec[{{coord3}}] += {{value}} * {{II.path_weight}} * L1_local_vec[{{coord1}}] * L2_local_vec[{{coord2}}];
                {%- endfor %}   
            }

            // WRITE TO GEMM_L1L2
 
            #pragma unroll 
            for(int L3_irrep_index = 0; L3_irrep_index < L3_irrep_length; L3_irrep_index++){
                gemm_L1L2_multipurpose_shared_warp[(tile.thread_rank() * L3_irrep_length) + L3_irrep_index] = active_for_tensor_product ? L1L2_multipurpose_local_vec[L3_irrep_index] : 0.0f; 
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

            #if 0
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
                {%- for i in range(II.tensor.nnz) %} 
                        {%- set coord1, coord2, coord3, value = II.tensor.tuples[i] %}
                        scratch[{{i % num_scratch_reg}}] = L1L2_multipurpose_local_vec[{{coord3}}] * ({{value}}*{{II.path_weight}}); 
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
}