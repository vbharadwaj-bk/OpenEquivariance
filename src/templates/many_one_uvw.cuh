{# Jinja2 Template #}

{% include 'common.cuh' %}
{%- from 'macros.jinja' import declare_smem_arrays, transpose_store with context %}
{%- from 'wmm.cuh' import generate_matmul %}

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

{{generate_matmul("warp_matmul", L3[0].mul, L3[0].ir.dim, L1[0].mul, 4, True)}}

__device__ __forceinline__ void forward_many_one(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem,
        const float* __restrict__ weights_shft,
        float* weights_smem, float* __restrict__ L3_smem, float* __restrict__ L3_scratch, int lane_id) {
    float l1_vec[{{L1_irrep_lengths | max}}];
    float l2_vec[{{L2_irrep_lengths | max}}];
    float l3_vec[{{L3_irrep_lengths | max}}];

    {%- set num_interact = interactions | length %}
    
    {%- for k in range(num_interact) %}
        {%- set u, v, w, instruction_idx, tensor = interactions[k] %}
        {%- set weight_start, _, _ = config.weight_range_and_shape_for_instruction(instruction_idx)%}

        for(int k = 0; k < {{L2[0].mul}}; k++) {

            if(lane_id < {{L1[u].mul}}) {
                #pragma unroll
                for(int j = 0; j < {{L1[u].ir.dim}}; j++)
                    l1_vec[j] = L1_smem[{{L1[u].ir.dim}} * lane_id + j];

                #pragma unroll
                for(int j = 0; j < {{L2[v].ir.dim}}; j++)
                    l2_vec[j] = L2_smem[k * {{L2[0].ir.dim}} + 
                        j + {{L2.slices()[v].start}}];

                #pragma unroll
                for(int j = 0; j < {{L3[w].ir.dim}}; j++)
                    l3_vec[j] = 0.0f;

                {%- for i in range(tensor.nnz) %}
                    {%- set coord1, coord2, coord3, value = tensor.tuples[i] %}
                    l3_vec[{{coord3}}] += {{value}} * l1_vec[{{coord1}}] * l2_vec[{{coord2}}];
                {%- endfor %}

                // Store to appropriate location, after transposing within warp, to shared memory 
                #pragma unroll
                for(int j = 0; j < {{L3[w].ir.dim}}; j++) {
                    L3_scratch[{{L3[w].ir.dim}} * lane_id + j] = l3_vec[j]; 
                }
            }

            ROW_OPERATION({{L3[0].mul * L1[0].mul}}, j, weights_smem[j + lane_id] = weights_shft[j + k * {{L3[0].mul * L1[0].mul}}];)
            __syncwarp();
            warp_matmul(weights_smem, L3_scratch, L3_smem);
            __syncwarp();
        }
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
            ("weights_smem", "float", 32 * 32)
        ]}, "warp_loc", forward_config)}}

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.dim}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3.dim}} + lane_id;
        float* weights_shft = weights + lane_id;

        ROW_OPERATION({{L1.dim}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        ROW_OPERATION({{L2.dim}}, j, L2_smem[j + lane_id] = l2_shft[j];)
        ROW_OPERATION({{L3.dim}}, j, L3_smem[j + lane_id] = 0.0f;)

        __syncwarp();

        forward_many_one(L1_smem, L2_smem, weights_shft, weights_smem, L3_smem, L3_scratch, lane_id); 

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
