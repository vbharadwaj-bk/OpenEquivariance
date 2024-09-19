{# Jinja2 Template #}

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE {{thead_block_size}}

// Assumes that L1 has multiplicity 32, L2 has multiplicity 1, L3 has multiplicity 32,
// column-major data layout
__global__ void loop_unroll_many_to_one(
    size_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, num_products);

    for(size_t i = start; i < end; i++) {
        float l1_vec[{{L1_one_rep_len}}];
        float l2_vec[{{L2_one_rep_len}}];
        float l3_vec[{{L3_one_rep_len}}];

        float* l1_start = L1_in + i * {{L1_stride}};
        float* l2_start = L2_in + i * {{L2_stride}};
        float* l3_start = L3_out + i * {{L3_stride}};

        #pragma unroll
        for(int j = 0; j < {{L1_one_rep_len}}; j++) {
            l1_vec[j] = l1_start[lane_id + {{L1_mult}} * j];
        }

        #pragma unroll
        for(int j = 0; j < {{L2_one_rep_len}}; j++) {
            l2_vec[j] = l2_start[j]; // All threads read common values for l2 
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
            l3_start[lane_id + {{L3_mult}} * j] = l3_vec[j];
        }
    }
}