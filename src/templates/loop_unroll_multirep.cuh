{# Jinja2 Template #}

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE {{thread_block_size}}
#define WARPS_PER_BLOCK THREAD_BLOCK_SIZE / THREADS_PER_WARP

// TODO: This pragma needs to be rewritten!
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

// Assumes all reps in L1 have multiplicity 32, all reps in L2 have multiplicity 1. 
// column-major data layout 
__global__ void loop_unroll_many_to_one(
    size_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

    __shared__ float L1_smem_full[WARPS_PER_BLOCK * {{L1_REP_LEN}}]; 
    __shared__ float L2_smem_full[WARPS_PER_BLOCK * {{L2_REP_LEN}}];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % (WARPS_PER_BLOCK);

    float* L1_smem = L1_smem_full + warp_loc * {{ L1_REP_LEN }};
    float* L2_smem = L2_smem_full + warp_loc * {{ L2_REP_LEN }};

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, num_products);

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1_REP_LEN}} + lane_id;
        float* l2_shft = L2_in + i * {{L2_REP_LEN}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3_REP_LEN}} + lane_id;

        ROW_OPERATION({{L1_REP_LEN}}, j,
            L1_smem[j + lane_id] = l1_shft[j];
        )

        ROW_OPERATION({{L2_REP_LEN}}, j,
            L2_smem[j + lane_id] = l2_shft[j];
        )

        float l1_vec[{{L1_one_rep_len}}];
        float l2_vec[{{L2_one_rep_len}}];
        float l3_vec[{{L3_one_rep_len}}];

        #pragma unroll
        for(int j = 0; j < {{L1_one_rep_len}}; j++) {
            l1_vec[j] = L1_smem[lane_id + {{L1_mult}} * j];
        }

        #pragma unroll
        for(int j = 0; j < {{L2_one_rep_len}}; j++) {
            l2_vec[j] = L2_smem[j];    
        }

        #pragma unroll
        for(int j = 0; j < {{L3_one_rep_len}}; j++) {
            l3_vec[j] = 0.0f; 
        }

        {# Value, L1_idx, L2_idx, L3_idx in each tuple #}
        {%- for i in range(nnz) %}
        l3_vec[{{coord3[i]}}] += {{values[i]}} * l1_vec[{{coord1[i]}}] * l2_vec[{{coord2[i]}}];  
        {%- endfor %}

        #pragma unroll
        for(int j = 0; j < {{L3_one_rep_len}}; j++) {
            l3_shft[{{L3_mult}} * j] = l3_vec[j];
        }
    }
}