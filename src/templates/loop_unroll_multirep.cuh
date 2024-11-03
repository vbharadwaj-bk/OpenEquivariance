{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import declare_smem_arrays with context %}

#define THREADS_PER_WARP {{ forward_config.warp_size }} // Warp size should be the same for forward and backward
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

__device__ __forceinline__ void forward_loop_unroll(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem, 
        const float* __restrict__ weights_smem, float* __restrict__ L3_smem) {
    float l1_vec[{{L1_irrep_lengths | max}}];
    float l2_vec[{{L2_irrep_lengths | max}}];
    float l3_vec[{{L3_irrep_lengths | max}}];
    float weight;

    {%- set num_interact = interactions | length %}
    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}

        weight = weights_smem[{{weights.offsets[k]}}];

        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            #pragma unroll
            for(int j = 0; j < {{L1[u].ir.dim}}; j++)
                l1_vec[j] = L1_smem[{{L1[u].mul}} * j + {{ L1.slices()[u].start}}];
        {%- endif %}

        {%- if k == 0 or interactions[k][1] != interactions[k-1][1] %}
            #pragma unroll
            for(int j = 0; j < {{L2[v].ir.dim}}; j++)
                l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}}];
        {%- endif %}

        {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
            #pragma unroll
            for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                l3_vec[j] = 0.0f;
        {%- endif %}

        {%- for i in range(tensor.nnz) %}
            {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
            l3_vec[{{coord3}}] += {{value}} * l1_vec[{{coord1}}] * l2_vec[{{coord2}}];
        {%- endfor %}

        // TODO: Should change to += accumulate, buffer the output in shared memory. 
        {%- if k == num_interact - 1 or interactions[k][2] != interactions[k+1][2] %}
            #pragma unroll
            for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                L3_smem[{{L3[w].mul}} * j + {{L3.slices()[w].start}}] = l3_vec[j] * weight;
        {%- endif %}
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
            ("weights_smem", "float", weights.total_len)
        ]}, "warp_loc", forward_config)}}

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.dim}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3.dim}} + lane_id;
        float* weights_shft = weights + i * {{weights.total_len}} + lane_id;

        ROW_OPERATION({{L1.dim}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        ROW_OPERATION({{L2.dim}}, j, L2_smem[j + lane_id] = l2_shft[j];)
        ROW_OPERATION({{L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)
        ROW_OPERATION({{weights.total_len}}, j, weights_smem[j + lane_id] = weights_shft[j];)

        __syncwarp();
        forward_loop_unroll(L1_smem + lane_id, L2_smem, weights_smem + lane_id, L3_smem + lane_id);
        __syncwarp();

        ROW_OPERATION({{L3.dim}}, j, l3_shft[j] = L3_smem[j + lane_id];)
    }
}

__device__ __forceinline__ void backward_loop_unroll(
        const float* L1_smem,
        const float* L2_smem,
        const float* weights_smem,
        const float* L3_grad_smem,

        float* L1_grad_smem,
        float* L2_grad_smem,
        float* weights_grad_smem,
        int lane_id) {

    float l1_vec[{{L1_irrep_lengths  | max}}]; 
    float l1_grad[{{L1_irrep_lengths | max}}]; 
    float l2_vec[{{L2_irrep_lengths  | max}}];
    float l2_grad[{{L2_irrep_lengths | max}}]; 
    float l3_grad[{{L3_irrep_lengths | max}}];

    float weight, weight_grad;

    {% set num_scratch_reg = 2 %}
    float scratch1[{{num_scratch_reg}}];
    float scratch2[{{num_scratch_reg}}];

    {%- set num_interact = interactions | length %}
    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        weight = weights_smem[{{weights.offsets[k]}}];
        weight_grad = weights_grad_smem[{{weights.offsets[k]}}];

        //==========================================
        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            #pragma unroll
            for(int j = 0; j < {{L1[u].ir.dim}}; j++) {
                l1_vec[j] = L1_smem[{{L1[u].mul}} * j + {{ L1.slices()[u].start}}];
                l1_grad[j] = L1_grad_smem[{{L1[u].mul}} * j + {{ L1.slices()[u].start}}];
            }
        {%- endif %}

        {%- if k == 0 or interactions[k][1] != interactions[k-1][1] %}
            #pragma unroll
            for(int j = 0; j < {{L2[v].ir.dim}}; j++) {
                l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}}];
                l2_grad[j] = L2_grad_smem[j + {{L2.slices()[v].start}}];
            }
        {%- endif %}

        {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
            #pragma unroll
            for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                l3_grad[j] = L3_grad_smem[{{L3[w].mul}} * j + {{ L3.slices()[w].start}}];
        {%- endif %}

        {%- for i in range(tensor.nnz) %} 
            {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
            scratch1[{{i % num_scratch_reg}}] = l3_grad[{{coord3}}] * {{value}}; 
            weight_grad += scratch1[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}] * l1_vec[{{coord1}}];
            scratch2[{{i % num_scratch_reg}}] = scratch1[{{i % num_scratch_reg}}] * weight;
            l2_grad[{{coord2}}] += scratch2[{{i % num_scratch_reg}}] * l1_vec[{{coord1}}];
            l1_grad[{{coord1}}] += scratch2[{{i % num_scratch_reg}}] * l2_vec[{{coord2}}];
        {%- endfor %}

        // Storeback
        {%- if k == num_interact - 1 or interactions[k][0] != interactions[k+1][0] %}
            #pragma unroll
            for(int j = 0; j < {{L1[u].ir.dim}}; j++)
                L1_grad_smem[{{L1[u].mul}} * j + {{ L1.slices()[u].start}}] = l1_grad[j];
        {%- endif %}

        {%- if k == num_interact - 1 or interactions[k][1] != interactions[k+1][1] %}
            // This assumes that all 32 threads are hit the same l2 vector. 
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                #pragma unroll
                for(int j = 0; j < {{L2[v].ir.dim}}; j++) {
                    l2_grad[j] += __shfl_down_sync(FULL_MASK, l2_grad[j], offset);
                } 
            }

            if(lane_id == 0) {
                #pragma unroll 
                for(int j = 0; j < {{L2[v].ir.dim}}; j++)
                    L2_grad_smem[j + {{L2.slices()[v].start}}] = l2_grad[j];
            }
            __syncwarp();
        {%- endif %}

        weights_grad_smem[{{weights.offsets[k]}}] = weight_grad; 
    {%- endfor %}
}

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
    float* L1_in, float* L1_grad,
    float* L2_in, float* L2_grad,
    float* weights, float* weights_grad,
    float* L3_grad) {

    {{ set_launch_bound_variables(backward_config) }}

    {{ declare_smem_arrays({
        "common": [],
        "per_warp": [
            ("L1_smem", "float", L1.dim),
            ("L1_grad_smem", "float", L1.dim),
            ("L2_smem", "float", L2.dim),
            ("L2_grad_smem", "float", L2.dim),
            ("weights_smem", "float", weights.total_len),
            ("weights_grad_smem", "float", weights.total_len),
            ("L3_grad_smem", "float", L3.dim)
        ]}, "warp_loc", backward_config)}}

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.dim}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3_shft = L3_grad + i * {{L3.dim}} + lane_id;
        float* weights_shft = weights + i * {{weights.total_len}} + lane_id;

        ROW_OPERATION({{L1.dim}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        ROW_OPERATION({{L2.dim}}, j, L2_smem[j + lane_id] = l2_shft[j];)
        ROW_OPERATION({{L3.dim}}, j, L3_grad_smem[j + lane_id] = l3_shft[j];)
        ROW_OPERATION({{weights.total_len}}, j, weights_smem[j + lane_id] = weights_shft[j];)

        ROW_OPERATION({{L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
        ROW_OPERATION({{L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
        ROW_OPERATION({{weights.total_len}}, j, weights_grad_smem[j + lane_id] = 0.0f;)

        __syncwarp();
        backward_loop_unroll(L1_smem + lane_id, L2_smem, weights_smem + lane_id, L3_grad_smem + lane_id,
                L1_grad_smem + lane_id, L2_grad_smem, weights_grad_smem + lane_id, lane_id);
        __syncwarp();

        float* l1_grad_shft = L1_grad + i * {{L1.dim}} + lane_id;
        float* l2_grad_shft = L2_grad + i * {{L2.dim}} + lane_id; 
        float* weights_grad_shft = weights_grad + i * {{weights.total_len}} + lane_id;

        ROW_OPERATION({{L1.dim}}, j, l1_grad_shft[j] = L1_grad_smem[j + lane_id];)
        ROW_OPERATION({{L2.dim}}, j, l2_grad_shft[j] = L2_grad_smem[j + lane_id];)
        ROW_OPERATION({{weights.total_len}}, j, weights_grad_shft[j] = weights_grad_smem[j + lane_id];)
    }
}