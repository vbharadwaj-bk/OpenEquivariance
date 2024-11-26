{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import declare_smem_arrays, transpose_load, transpose_store with context %}
{%- from 'macros.jinja' import load_ir_segments, store_ir_segments, declare_smem_variables with context %}

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

{%- include 'loop_unroll_tp.cuh' %}
{%- from 'loop_unroll_tp.cuh' import generate_segment_kernel_forward %}

{%- macro generate_segment_kernel_backward(id, segment) %}
{%- set L1, L2, L3, interactions, problem = segment.L1, segment.L2, segment.L3, segment.interactions, segment.problem %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

__device__ __forceinline__ void backward_loop_unroll_{{id}}(
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
    int offset;

    {%- set num_scratch_reg = 1 %}
    float scratch1[{{num_scratch_reg}}];
    float scratch2[{{num_scratch_reg}}];

    {%- set num_interact = interactions | length %}

    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        {%- set weight_start, _, _ = config.weight_range_and_shape_for_instruction(k)%}
        weight = weights_smem[{{weight_start}}];
        weight_grad = weights_grad_smem[{{weight_start}}];

        {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
            offset = {{ L1.slices()[u].start}};
            {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_smem', 'offset', 'l1_vec')}}
            {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_grad_smem', 'offset', 'l1_grad')}}
        {%- endif %}

        {%- if k == 0 or interactions[k][1] != interactions[k-1][1] %}
            #pragma unroll
            for(int j = 0; j < {{L2[v].ir.dim}}; j++) {
                l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}}];
                l2_grad[j] = 0.0; 
            }
        {%- endif %}

        {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
            offset = {{ L3.slices()[w].start}}; 
            {{transpose_load(L3[w].mul, L3[w].ir.dim, 'L3_grad_smem', 'offset', 'l3_grad')}}
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
            offset = {{ L1.slices()[u].start}}; 
            {{transpose_store(L1[u].mul, L1[u].ir.dim, 'L1_grad_smem', 'offset', 'l1_grad', '=', '1.0')}}
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
                    L2_grad_smem[j + {{L2.slices()[v].start}}] += l2_grad[j];
            }
            __syncwarp();
        {%- endif %}

        weights_grad_smem[{{weight_start}}] = weight_grad; 
    {%- endfor %}
}
{%- endmacro %}

{%- for i, segment in enumerate(forward_schedule.segments) %}
{{ generate_segment_kernel_forward(i, segment) }}
{%- endfor %}

{%- for i, segment in enumerate(backward_schedule.segments) %}
{{ generate_segment_kernel_backward(i, segment) }}
{%- endfor %}

__global__ void forward(
        size_t num_products, float* L1_in, float* L2_in, float* L3_out, float* weights) {
    extern __shared__ char s[];
    {{ set_launch_bound_variables(forward_config) }}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        float* l1 = L1_in + i * {{L1.dim}} + lane_id;
        float* l2 = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3 = L3_out + i * {{L3.dim}} + lane_id;
        float* w = weights + i * {{config.weight_numel}};

        {%- for i, segment in enumerate(forward_schedule.segments) %} {
            {{ declare_smem_variables(segment, "smem") }}
            {{ load_ir_segments(segment.L1Map, "l1", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2", "L2_smem", "j") }}
            ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w[{{segment.weight_offset}} + j + lane_id];)

            __syncwarp();
            forward_loop_unroll_{{i}}(L1_smem, L2_smem, weights_smem + lane_id, L3_smem, lane_id);
            __syncwarp();

            {{ store_ir_segments(segment.L3Map, "l3", "L3_smem", "j") }}
        } {%- endfor %}
    }
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
    extern __shared__ char s[];
    {{ set_launch_bound_variables(backward_config) }}
    char* smem = s + {{backward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.dim}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3_shft = L3_grad + i * {{L3.dim}} + lane_id;
        float* weights_shft = weights + i * {{config.weight_numel}} + lane_id;

        {%- for i, segment in enumerate(backward_schedule.segments) %} {
            {{ declare_smem_variables(segment, "smem") }}

            {{ load_ir_segments(segment.L1Map, "l1_shft", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2_shft", "L2_smem", "j") }}
            {{ load_ir_segments(segment.L3Map, "l3_shft", "L3_grad_smem", "j") }}
            ROW_OPERATION({{config.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[j];)

            ROW_OPERATION({{L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{config.weight_numel}}, j, weights_grad_smem[j + lane_id] = 0.0f;)

            __syncwarp();
            backward_loop_unroll_{{i}}(L1_smem, L2_smem, weights_smem + lane_id, L3_grad_smem,
                    L1_grad_smem, L2_grad_smem, weights_grad_smem + lane_id, lane_id);
            __syncwarp();

            float* l1_grad_shft = L1_grad + i * {{L1.dim}} + lane_id;
            float* l2_grad_shft = L2_grad + i * {{L2.dim}} + lane_id; 
            float* weights_grad_shft = weights_grad + i * {{config.weight_numel}} + lane_id;

            {{ store_ir_segments(segment.L1Map, "l1_grad_shft", "L1_grad_smem", "j") }}
            {{ store_ir_segments(segment.L2Map, "l2_grad_shft", "L2_grad_smem", "j") }}

            ROW_OPERATION({{config.weight_numel}}, j, weights_grad_shft[j] = weights_grad_smem[j + lane_id];)
        } {%- endfor %}
    }
}