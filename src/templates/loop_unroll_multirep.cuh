{# Jinja2 Template #}

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE {{thread_block_size}}
#define WARPS_PER_BLOCK THREAD_BLOCK_SIZE / THREADS_PER_WARP

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

    __shared__ float L1_smem_full[WARPS_PER_BLOCK * {{L1.rep_len}}]; 
    __shared__ float L2_smem_full[WARPS_PER_BLOCK * {{L2.rep_len}}];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % (WARPS_PER_BLOCK);

    float* L1_smem = L1_smem_full + warp_loc * {{ L1.rep_len }};
    float* L2_smem = L2_smem_full + warp_loc * {{ L2.rep_len }};

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, num_products);

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.rep_len}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.rep_len}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3.rep_len}} + lane_id;

        ROW_OPERATION({{L1.rep_len}}, j,
            L1_smem[j + lane_id] = l1_shft[j];
        )

        ROW_OPERATION({{L2.rep_len}}, j,
            L2_smem[j + lane_id] = l2_shft[j];
        )

        {%- for u, v, w, tensor in interactions %}
        {
            float l1_vec[{{L1.irrep_lengths[u]}}];
            float l2_vec[{{L2.irrep_lengths[v]}}];
            float l3_vec[{{L3.irrep_lengths[w]}}];

            #pragma unroll
            for(int j = 0; j < {{L1.irrep_lengths[u]}}; j++) {
                l1_vec[j] = L1_smem[lane_id + {{L1.mults[u]}} * j + {{ L1.offsets[u]}}];
            }

            #pragma unroll
            for(int j = 0; j < {{L2.irrep_lengths[v]}}; j++) {
                l2_vec[j] = L2_smem[j + {{L2.offsets[v]}}];
            }

            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++) {
                l3_vec[j] = 0.0f;
            }

            {# Value, L1_idx, L2_idx, L3_idx in each tuple #}
            {%- for i in range(tensor.nnz) %}
                l3_vec[{{tensor.coord3[i]}}] += {{tensor.values[i]}} * l1_vec[{{tensor.coord1[i]}}] * l2_vec[{{tensor.coord2[i]}}];
            {%- endfor %}

            // TODO: Should change to += accumulate, buffer the output in shared memory. 
            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++) {
                l3_shft[{{L3.mults[w]}} * j + {{L3.offsets[w]}}] = l3_vec[j];
            }
        }
        {%- endfor %}
    }
}