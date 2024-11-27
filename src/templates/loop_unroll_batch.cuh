{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import declare_smem_arrays, transpose_load, transpose_store with context %}
{%- from 'macros.jinja' import load_ir_segments, store_ir_segments, declare_smem_variables with context %}

{%- include 'loop_unroll_tp.cuh' %}
{%- from 'loop_unroll_tp.cuh' import generate_segment_kernel_forward, generate_segment_kernel_backward %}

#define THREADS_PER_WARP {{ forward_schedule.launch_config.warp_size }} // Warp size should be the same for forward and backward
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

{%- for i, segment in enumerate(forward_schedule.segments) %}
{{ generate_segment_kernel_forward(i, segment) }}
{%- endfor %}

__global__ void forward(
        size_t num_products, float* L1_in, float* L2_in, float* L3_out, float* weights) {
    extern __shared__ char s[];
    {{ set_launch_bound_variables(forward_schedule.launch_config) }}
    {%- set tpp = forward_schedule.updated_config %}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        float* l1 = L1_in + i * {{forward_schedule.L1.dim}} + lane_id;
        float* l2 = L2_in + i * {{forward_schedule.L2.dim}} + lane_id; 
        float* l3 = L3_out + i * {{forward_schedule.L3.dim}} + lane_id;
        float* w = weights + i * {{tpp.weight_numel}};

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

{%- for i, segment in enumerate(backward_schedule.segments) %}
{{ generate_segment_kernel_backward(i, segment) }}
{%- endfor %}

__global__ void backward(
        size_t num_products,
        float* L1_in, float* L1_grad,
        float* L2_in, float* L2_grad,
        float* weights, float* weights_grad,
        float* L3_grad) {
    extern __shared__ char s[];
    {{ set_launch_bound_variables(backward_schedule.launch_config) }}
    char* smem = s + {{backward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        {%- set tpp = backward_schedule.updated_config %}
        float* l1_shft = L1_in + i * {{backward_schedule.L1.dim}} + lane_id;
        float* l2_shft = L2_in + i * {{backward_schedule.L2.dim}} + lane_id; 
        float* l3_shft = L3_grad + i * {{backward_schedule.L3.dim}} + lane_id;
        float* weights_shft = weights + i * {{tpp.weight_numel}} + lane_id;

        {%- for i, segment in enumerate(backward_schedule.segments) %} {
            {{ declare_smem_variables(segment, "smem") }}

            {{ load_ir_segments(segment.L1Map, "l1_shft", "L1_smem", "j") }}
            {{ load_ir_segments(segment.L2Map, "l2_shft", "L2_smem", "j") }}
            {{ load_ir_segments(segment.L3Map, "l3_shft", "L3_grad_smem", "j") }}
            ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = weights_shft[{{segment.weight_offset}} + j];)

            ROW_OPERATION({{segment.L1.dim}}, j, L1_grad_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{segment.L2.dim}}, j, L2_grad_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_smem[j + lane_id] = 0.0;)

            __syncwarp();
            backward_loop_unroll_{{i}}(L1_smem, L2_smem, weights_smem + lane_id, L3_grad_smem,
                    L1_grad_smem, L2_grad_smem, weights_grad_smem + lane_id, lane_id);
            __syncwarp();

            float* l1_grad_shft = L1_grad + i * {{backward_schedule.L1.dim}} + lane_id;
            float* l2_grad_shft = L2_grad + i * {{backward_schedule.L2.dim}} + lane_id;
            float* weights_grad_shft = weights_grad + i * {{backward_schedule.updated_config.weight_numel}} + lane_id;

            {{ store_ir_segments(segment.L1Map, "l1_grad_shft", "L1_grad_smem", "j") }}
            {{ store_ir_segments(segment.L2Map, "l2_grad_shft", "L2_grad_smem", "j") }}
            ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_grad_shft[{{segment.weight_offset}} + j] = weights_grad_smem[j + lane_id];)
        } {%- endfor %}
    }
}