{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import
        transpose_load, transpose_store, 
        load_ir_segments, load_ir_async, store_ir_segments, 
        declare_smem_variables,
        set_launch_bound_variables with context %}

{%- from 'loop_unroll_tp.cuh' import 
        generate_segment_kernel_forward, 
        generate_segment_kernel_backward %}

#include "cuda_pipeline_primitives.h"

#define THREADS_PER_WARP {{ forward_schedule.launch_config.warp_size }} // Warp size should be the same for forward and backward
#define FULL_MASK 0xffffffff

{%- for i, segment in enumerate(forward_schedule.segments) %}
{{ generate_segment_kernel_forward(i, segment) }}
{%- endfor %}

template <unsigned int N>
struct _Segment {
    int _seg[N];
};

// Trivial layout guaranteed-aligned copy-async compatible segments
template <unsigned int N>
struct Segment;
template <>
struct __align__(4) Segment<1> : public _Segment<1>{};
template <>
struct __align__(8) Segment<2> : public _Segment<2>{};
template <>
struct __align__(16) Segment<4> : public _Segment<4>{};


__global__ void forward(
        size_t num_products, float* L1_in, float* L2_in, float* L3_out, float* weights) {
    extern __shared__ char s[];
    {{ set_launch_bound_variables(forward_schedule.launch_config) }}
    {%- set tpp = forward_schedule.updated_config %}
    char* smem = s + {{forward_schedule.memory_per_warp}} * warp_loc; 

    for(size_t i = start; i < end; i++) {
        float* l1 = L1_in + i * {{forward_schedule.L1.dim}};
        float* l2 = L2_in + i * {{forward_schedule.L2.dim}} + lane_id; 
        float* l3 = L3_out + i * {{forward_schedule.L3.dim}} + lane_id;
        float* w = weights + i * {{tpp.weight_numel}};

        {%- for i, segment in enumerate(forward_schedule.segments) %} {
            {{ declare_smem_variables(segment, "smem") }}

            /*
            {{ load_ir_segments(segment.L1Map, "l1", "L1_smem", "j") }}
            */

            /*ROW_OPERATION({{segment.L1.dim // 4 }}, j, 
                    __pipeline_memcpy_async(L1_smem_group + lane_id + j, l1_group + j, 16, 0);)
            */

            {{ load_ir_async(segment.L1Map, "l1", "L1_smem", "j") }}
            __pipeline_commit();


            {{ load_ir_segments(segment.L2Map, "l2", "L2_smem", "j") }}
            ROW_OPERATION({{segment.L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)
            ROW_OPERATION({{segment.problem.weight_numel}}, j, weights_smem[j + lane_id] = w[{{segment.weight_offset}} + j + lane_id];)

            __pipeline_wait_prior(0);

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