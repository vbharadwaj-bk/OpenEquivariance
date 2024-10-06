{# Jinja2 Template #}

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE {{thread_block_size}}
#define WARPS_PER_BLOCK THREAD_BLOCK_SIZE / THREADS_PER_WARP

#define ROW_OPERATION(ROW_LEN, LOOP_VAR, ...) \
    _Pragma ("unroll") \
    for(int LOOP_VAR = 0; LOOP_VAR < ROW_LEN; LOOP_VAR += THREADS_PER_WARP) { \
        if(LOOP_VAR >= ROW_LEN - THREADS_PER_WARP) { \
            if(lane_id < ROW_LEN - LOOP_VAR) { \
                __VA_ARGS__  \
            } \
        } \
        else { \
            __VA_ARGS__  \
        } \
    }

{% macro smem_array(name, dtype, num_elements) -%}
    {{dtype}}* {{name}} = ({{dtype}}*) (s + {{ ns["offset"] }}); 
    {% do ns.update({"offset": ns["offset"] + " + " + num_elements + " * sizeof({})".format(dtype)}) %}
{%- endmacro %}

// Assumes all reps in L1 have multiplicity 32, all reps in L2 have multiplicity 1. 
// column-major data layout 
__global__ void loop_unroll_many_to_one(
    size_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

    extern __shared__ char s[];
    {% set ns = {"offset": "0"} %}
    {{ smem_array("L1_smem_full", "float", "WARPS_PER_BLOCK * " + L1.rep_len|string)}}
    {{ smem_array("L2_smem_full", "float", "WARPS_PER_BLOCK * " + L2.rep_len|string)}}
    {{ smem_array("L3_smem_full", "float", "WARPS_PER_BLOCK * " + L3.rep_len|string)}}

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % (WARPS_PER_BLOCK);

    float* L1_smem = L1_smem_full + warp_loc * {{ L1.rep_len }};
    float* L2_smem = L2_smem_full + warp_loc * {{ L2.rep_len }};
    float* L3_smem = L3_smem_full + warp_loc * {{ L3.rep_len }};

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, num_products);

    for(size_t i = start; i < end; i++) {
        float* l1_shft = L1_in + i * {{L1.rep_len}} + lane_id;
        float* l2_shft = L2_in + i * {{L2.rep_len}} + lane_id; 
        float* l3_shft = L3_out + i * {{L3.rep_len}} + lane_id;

        ROW_OPERATION({{L1.rep_len}}, j,
            L1_smem[j + lane_id] = l1_shft[j];
        )

        ROW_OPERATION({{L2.rep_len}}, j,
            L2_smem[j + lane_id] = l2_shft[j];
        )

        ROW_OPERATION({{L3.rep_len}}, j,
            L3_smem[j + lane_id] = 0.0f; 
        )

        __syncwarp();

        {% set num_interact = interactions | length %}
        {%- for k in range(num_interact) %}
            {% set u, v, w, tensor = interactions[k] %}
            {% set u_change = k < num_interact - 1 and interactions[k][0] != interactions[k+1][0] %}
            {% set v_change = k < num_interact - 1 and interactions[k][1] != interactions[k+1][1] %}
            {% set w_change = k < num_interact - 1 and interactions[k][2] != interactions[k+1][2] %}

            { 
            float l1_vec[{{L1.irrep_lengths[u]}}];
            #pragma unroll
            for(int j = 0; j < {{L1.irrep_lengths[u]}}; j++) {
                l1_vec[j] = L1_smem[lane_id + {{L1.mults[u]}} * j + {{ L1.offsets[u]}}];
            }

            float l2_vec[{{L2.irrep_lengths[v]}}];
            #pragma unroll
            for(int j = 0; j < {{L2.irrep_lengths[v]}}; j++) {
                l2_vec[j] = L2_smem[j + {{L2.offsets[v]}}];
            }

            // TODO: Should change to += accumulate, buffer the output in shared memory. 
            float l3_vec[{{L3.irrep_lengths[w]}}];
            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++) {
                l3_vec[j] = 0.0f;
            }

            {# Value, L1_idx, L2_idx, L3_idx in each tuple #}
            {%- for i in range(tensor.nnz) %}
                l3_vec[{{tensor.coord3[i]}}] += {{tensor.values[i]}} * l1_vec[{{tensor.coord1[i]}}] * l2_vec[{{tensor.coord2[i]}}];
            {%- endfor %}

            #pragma unroll
            for(int j = 0; j < {{L3.irrep_lengths[w]}}; j++) {
                L3_smem[lane_id + {{L3.mults[w]}} * j + {{L3.offsets[w]}}] = l3_vec[j];
            }
            }
        {%- endfor %}

        __syncwarp();

        ROW_OPERATION({{L3.rep_len}}, j,
            l3_shft[j] = L3_smem[j + lane_id]; 
        )
    }
}