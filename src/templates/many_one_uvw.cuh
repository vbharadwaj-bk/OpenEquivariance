{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import declare_smem_arrays, transpose_store with context %}

#define THREADS_PER_WARP {{ forward_config.warp_size }} 
#define FULL_MASK 0xffffffff

{%- macro set_launch_bound_variables(config) %}
    {%- set warps_per_block = divide(config.num_threads, config.warp_size) %}
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = t_idx / THREADS_PER_WARP;
    int lane_id = t_idx % THREADS_PER_WARP;
    int warp_loc = warp_id % {{ warps_per_block }};
    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, num_products);
{%- endmacro %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

/*
* Two template params: THREAD_ROWS, THREAD_COLS. Threads laid out in an 4 x 8 grid,
* so the output has dimensions (4 * THREAD_ROWS) x (8 * THREAD_COLS). 0-padding
* is applied to the input irreps but NOT the output buffer, conditionals are used for loads.
* Weights is N x K, irreps is M x K, output is M x N. 
*/

/*
template<int THREAD_ROWS, int THREAD_COLS, int M, int N, int K>
__device__ __forceinline__ warp_matmul(
            const float* __restrict__ irreps, 
            const float* __restrict__ weights, 
            float* __restrict__ out,
            int lane_id) {

    float output[THREAD_ROWS][THREAD_COLS];
    float col_weights[THREAD_COLS];
    float row_irreps[THREAD_ROWS];

    int t_row_idx = lane_id % 4; 
    int t_col_idx = lane_id / 4;

    int row_start = t_row_idx * THREAD_ROWS;
    int col_start = t_col_idx * THREAD_COLS;

    // Zero out the output buffer
    #pragma unroll
    for(int i = 0; i < THREAD_ROWS; i++) {
        #pragma unroll
        for(int j = 0; j < THREAD_COLS; j++)
            output[i][j] = 0.0f;
    }

    for(int k = 0; k < K; k++) {
        #pragma unroll
        for(int i = 0; i < THREAD_ROWS; i++)
            row_irreps[i] = irreps[row_start + i + k * M];

        #pragma unroll
        for(int j = 0; j < THREAD_COLS; j++)
            col_weights[j] = weights[col_start + j + k * N];

        #pragma unroll
        for(int i = 0; i < THREAD_ROWS; i++) {
            #pragma unroll
            for(int j = 0; j < THREAD_COLS; j++)
                output[i][j] += row_irreps[i] * col_weights[j];
        }
    }

    // This could be optimized to avoid the padding issue, or by writing
    // the arguments directly back to registers
    #pragma unroll
    for(int i = 0; i < THREAD_ROWS; i++) {
        #pragma unroll
        for(int j = 0; j < THREAD_COLS; j++) {
            if(row_start + i < M && col_start + j < N)
                out[(row_start + i) * N + col_start + j] = output[i][j];
        }
    }
}
*/

__device__ __forceinline__ void forward_many_one(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem, 
        const float* __restrict__ weights_smem, float* __restrict__ L3_smem) {
    float l1_vec[{{L1_irrep_lengths | max}}];
    float l2_vec[{{L2_irrep_lengths | max}}];
    float l3_vec[{{L3_irrep_lengths | max}}];
    float l3_vec_weighted[{{L3_irrep_lengths | max}}];

    // Assumes the weight matrix is 32 x 32.
    // Each thread stores a 32-length row of the weight matrix
    float weights[{{ forward_config.warp_size }}];

    {%- set num_interact = interactions | length %}
    
    {%- for k in range(num_interact) %}
        {%- set u, v, w, instruction_idx, tensor = interactions[k] %}
        {%- set weight_start, _, _ = config.weight_range_and_shape_for_instruction(instruction_idx)%}

        #pragma unroll
        for(int j = 0; j < {{ forward_config.warp_size }}; j++)
            weights[j] = weights_smem[{{weight_start}} + j * {{forward_config.warp_size}}];

        #pragma unroll
        for(int j = 0; j < {{L1[u].ir.dim}}; j++)
            l1_vec[j] = L1_smem[{{L1[u].mul}} * j + {{ L1.slices()[u].start}}];

        #pragma unroll
        for(int j = 0; j < {{L2[v].ir.dim}}; j++)
            l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}}];

        #pragma unroll
        for(int j = 0; j < {{L3[w].ir.dim}}; j++)
            l3_vec[j] = 0.0f;

        {%- for i in range(tensor.nnz) %}
            {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
            l3_vec[{{coord3}}] += {{value}} * l1_vec[{{coord1}}] * l2_vec[{{coord2}}];
        {%- endfor %}

        #pragma unroll
        for(int j = 0; j < {{L3[w].ir.dim}}; j++)
            l3_vec_weighted[j] = 0.0; 

        {%- for i in range(forward_config.warp_size) %}
            {%- for j in range(L3[w].ir.dim) %}
                l3_vec_weighted[{{j}}] += __shfl_sync(FULL_MASK, l3_vec[{{j}}], {{i}}) * weights[{{i}}];
            {%- endfor %}
        {%- endfor %}

        #pragma unroll
        for(int j = 0; j < {{L3[w].ir.dim}}; j++)
            L3_smem[{{L3[w].mul}} * j + {{L3.slices()[w].start}}] += l3_vec_weighted[j]; 
    {%- endfor %}
}

// Assumes all reps in L1 have multiplicity 32, all reps in L2 have multiplicity 1. 
// column-major data layout 
__global__ void forward(
    size_t num_products, float* L1_in, float* L2_in, float* L3_out, float* weights) {

    {{ set_launch_bound_variables(forward_config) }}

    {{ declare_smem_arrays({
        "common": [],
        "per_warp": [
            ("L1_smem", "float", L1.dim),
            ("L2_smem", "float", L2.dim),
            ("L3_smem", "float", L3.dim),
            ("L3_scratch", "float", L3.dim),
            ("weights_smem", "float", config.weight_numel)
        ]}, "warp_loc", forward_config)}}

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.dim}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3.dim}} + lane_id;
        float* weights_shft = weights + lane_id;

        ROW_OPERATION({{L1.dim}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        ROW_OPERATION({{L2.dim}}, j, L2_smem[j + lane_id] = l2_shft[j];)
        ROW_OPERATION({{L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)
        ROW_OPERATION({{config.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[j];)

        __syncwarp();
        forward_many_one(L1_smem + lane_id, L2_smem, weights_smem + lane_id, L3_smem + lane_id);
        __syncwarp();

        ROW_OPERATION({{L3.dim}}, j, l3_shft[j] = L3_smem[j + lane_id];)
    }
}

__global__ void backward(
    size_t num_products,
    float* L1_in, float* L1_grad,
    float* L2_in, float* L2_grad,
    float* weights, float* weights_grad,
    float* L3_grad) {
    
    {{ set_launch_bound_variables(backward_config) }}

    if(t_idx == 0) {
        printf("Backward kernel not implemented yet\n");
    }
}
