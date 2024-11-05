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
    size_t nnz_per_warp = (c.nnz + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, c.nnz);
{%- endmacro %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

struct ConvData {
    unsigned int* rows;
    unsigned int* cols;
    unsigned long nnz;
    unsigned int node_count;
};

/*
* Forward kernel assumes that rows, cols in ConvData sorted in row-major order.
*/
__global__ void forward(
        float* L1_in,
        float* L2_in,
        float* weights,
        float* L3_out,
        ConvData c,
        bool disable_tensor_op) {

    {{set_launch_bound_variables(forward_config)}}

    {{ declare_smem_arrays({
        "common": [],
        "per_warp": [
            ("L1_smem", "float", L1.dim),
            ("L2_smem", "float", L2.dim),
            ("L3_smem", "float", L3.dim),
            ("weights_smem", "float", config.weight_numel)
        ]}, "warp_loc", forward_config)}}

    bool firstSegment = true;

    ROW_OPERATION({{L3.dim}}, j, L3_smem[j + lane_id] = 0.0;)

    for(size_t i = start; i < end; i++) {
        size_t row = c.rows[i];
        size_t col = c.cols[i];

        float* l1_shft = L1_in + col * {{L1.dim}} + lane_id;
        //float* l2_shft = L2_in + i * {{L2.dim}} + lane_id; 
        float* l3_shft = L3_out + row * {{L3.dim}} + lane_id;

        ROW_OPERATION({{L1.dim}}, j, L1_smem[j + lane_id] = l1_shft[j];)
        //__syncwarp();
        if(! disable_tensor_op) {

        }
        __syncwarp();
        ROW_OPERATION({{L3.dim}}, j, L3_smem[j + lane_id] += L1_smem[j + lane_id];)

        bool changeRow = (i < end - 1) && (row != c.rows[i+1]);

        // If changing rows and this is not the first segment or the last segment, global write
        if(changeRow && ! firstSegment) {
            ROW_OPERATION({{L3.dim}}, j,
                l3_shft[j] += L3_smem[j + lane_id];
                L3_smem[j + lane_id] = 0.0; // Zero out buffer for next accumulation
            )
        }
        // If this is either the first segment (and changing row) or the last segment, atomicAdd (fixup)
        else if(i == end - 1 || (firstSegment && changeRow) ) {
            ROW_OPERATION({{L3.dim}}, j,
                atomicAdd(l3_shft + j, L3_smem[j + lane_id]);
                L3_smem[j + lane_id] = 0.0; 
            )
            firstSegment = false;
        }
    }
}