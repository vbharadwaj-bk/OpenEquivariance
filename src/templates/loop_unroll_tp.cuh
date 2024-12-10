{%- from 'macros.jinja' import transpose_load, transpose_store with context %}

{%- macro generate_segment_kernel_forward(id, segment) %}
{%- set L1, L2, L3, interactions, problem = segment.L1, segment.L2, segment.L3, segment.interactions, segment.problem %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

__device__ __forceinline__ void forward_loop_unroll_{{id}}(const IRREP_T* __restrict__ L1_smem, const IRREP_T* __restrict__ L2_smem, 
        const WEIGHT_T* __restrict__ weights_smem, IRREP_T* __restrict__ L3_smem, int lane_id) {
    IRREP_T l1_vec[{{L1_irrep_lengths | max}}];
    IRREP_T l2_vec[{{L2_irrep_lengths | max}}];
    IRREP_T l3_vec[{{L3_irrep_lengths | max}}];
    WEIGHT_T weight;
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

{%- macro generate_segment_kernel_backward(id, segment) %}
{%- set L1, L2, L3, interactions, problem = segment.L1, segment.L2, segment.L3, segment.interactions, segment.problem %}

{%- set L1_irrep_lengths = L1 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L2_irrep_lengths = L2 | map(attribute="ir") | map(attribute="dim") | list %}
{%- set L3_irrep_lengths = L3 | map(attribute="ir") | map(attribute="dim") | list %}

__device__ __forceinline__ void backward_loop_unroll_{{id}}(
        const IRREP_T* L1_smem,
        const IRREP_T* L2_smem,
        const WEIGHT_T* weights_smem,
        const IRREP_T* L3_grad_smem,

        IRREP_T* L1_grad_smem,
        IRREP_T* L2_grad_smem,
        WEIGHT_T* weights_grad_smem,
        int lane_id) {

    IRREP_T l1_vec[{{L1_irrep_lengths  | max}}]; 
    IRREP_T l1_grad[{{L1_irrep_lengths | max}}]; 
    IRREP_T l2_vec[{{L2_irrep_lengths  | max}}];
    IRREP_T l2_grad[{{L2_irrep_lengths | max}}]; 
    IRREP_T l3_grad[{{L3_irrep_lengths | max}}];

    WEIGHT_T weight, weight_grad;
    int offset;

    {%- set num_scratch_reg = 1 %}
    IRREP_T scratch1[{{num_scratch_reg}}];
    IRREP_T scratch2[{{num_scratch_reg}}];

    {%- set num_interact = interactions | length %}

    {%- for k in range(num_interact) %}
        {%- set u, v, w, tensor = interactions[k] %}
        {%- set weight_start, _, _ = problem.weight_range_and_shape_for_instruction(k)%}
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