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

struct ConvData {
    uint32_t* rows;
    uint32_t* cols;
    uint64_t nnz;
    uint32_t node_count;
};

__global__ void forward(
        float* L1_in,
        float* L2_in,
        float* weights,
        float* L3_out,
        ConvData conv_data,
        bool disable_tensor_op) {

    set_launch_bound_variables(forward_config);

    if(t_idx == 0) {
        printf("Hello world from inside kernel!");
    }
}