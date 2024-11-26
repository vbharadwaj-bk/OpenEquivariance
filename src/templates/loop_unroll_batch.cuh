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

{% include 'loop_unroll_tp.cuh' %}

{%- macro generate_segment_kernel(id, segment) %}
{%- set L1, L2, L3, interactions, problem = segment.L1, segment.L2, segment.L3, segment.interactions, segment.problem %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

__device__ __forceinline__ void forward_loop_unroll_{{id}}(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem, 
        const float* __restrict__ weights_smem, float* __restrict__ L3_smem, int lane_id) {
    float l1_vec[{{L1_irrep_lengths | max}}];
    float l2_vec[{{L2_irrep_lengths | max}}];
    float l3_vec[{{L3_irrep_lengths | max}}];
    float weight;
    int offset;

    {%- set num_interact = interactions | length %}
    
    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        {%- set weight_start, _, _ = problem.weight_range_and_shape_for_instruction(k)%}

        if(lane_id < {{L1[u].mul}}) {
            weight = weights_smem[{{weight_start}}];

            {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
                offset = {{ L1.slices()[u].start}}; 
                {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_smem', 'offset', 'l1_vec')}}
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

            {%- if k == num_interact - 1 or interactions[k][2] != interactions[k+1][2] %}
                offset = {{ L3.slices()[w].start}}; 
                {{transpose_store(L3[w].mul, L3[w].ir.dim, 'L3_smem', 'offset', 'l3_vec', '+=', 'weight')}}
            {%- endif %}
        }
    {%- endfor %}
}
{%- endmacro %}

{%- for i, segment in enumerate(forward_schedule.segments) %}
{{ generate_segment_kernel(i, segment) }}
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
            ROW_OPERATION({{L2.dim}}, j, L2_smem[j + lane_id] = l2_shft[j];)
            ROW_OPERATION({{L3.dim}}, j, L3_grad_smem[j + lane_id] = l3_shft[j];)
            ROW_OPERATION({{config.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[j];)

            ROW_OPERATION({{L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{config.weight_numel}}, j, weights_grad_smem[j + lane_id] = 0.0f;)

            __syncwarp();
            backward_loop_unroll(L1_smem, L2_smem, weights_smem + lane_id, L3_grad_smem,
                    L1_grad_smem, L2_grad_smem, weights_grad_smem + lane_id, lane_id);
            __syncwarp();

            float* l1_grad_shft = L1_grad + i * {{L1.dim}} + lane_id;
            float* l2_grad_shft = L2_grad + i * {{L2.dim}} + lane_id; 
            float* weights_grad_shft = weights_grad + i * {{config.weight_numel}} + lane_id;

            ROW_OPERATION({{L1.dim}}, j, l1_grad_shft[j] = L1_grad_smem[j + lane_id];)
            ROW_OPERATION({{L2.dim}}, j, l2_grad_shft[j] = L2_grad_smem[j + lane_id];)
            ROW_OPERATION({{config.weight_numel}}, j, weights_grad_shft[j] = weights_grad_smem[j + lane_id];)
        } {%- endfor %}
    }
}