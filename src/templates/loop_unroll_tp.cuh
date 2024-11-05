__device__ __forceinline__ void forward_loop_unroll(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem, 
        const float* __restrict__ weights_smem, float* __restrict__ L3_smem) {
    float l1_vec[{{L1_irrep_lengths | max}}];
    float l2_vec[{{L2_irrep_lengths | max}}];
    float l3_vec[{{L3_irrep_lengths | max}}];
    float weight;

    {%- set num_interact = interactions | length %}
    {%- for k in range(num_interact) %}
        {%- set u, v, w, instruction_idx, tensor = interactions[k] %}
        {%- set weight_start, _, _ = config.weight_range_and_shape_for_instruction(instruction_idx)%}

        weight = weights_smem[{{weight_start}}];

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