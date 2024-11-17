{%- from 'macros.jinja' import transpose_load, transpose_store with context %}

__device__ __forceinline__ void forward_loop_unroll(const float* __restrict__ L1_smem, const float* __restrict__ L2_smem, 
        const float* __restrict__ weights_smem, float* __restrict__ L3_smem, int lane_id) {
    float l1_vec[{{L1_irrep_lengths | max}}];
    float l2_vec[{{L2_irrep_lengths | max}}];
    float l3_vec[{{L3_irrep_lengths | max}}];
    float weight;
    int offset;

    {%- set num_interact = interactions | length %}
    
    for(int mul_offset = 0; mul_offset < {{L1[0].mul}}; mul_offset += {{forward_config.warp_size}}) {
        if(mul_offset + lane_id < {{L1[0].mul}}) {
            {%- for k in range(num_interact) %}
                {%- set u, v, w, instruction_idx, tensor = interactions[k] %}
                {%- set weight_start, _, _ = config.weight_range_and_shape_for_instruction(instruction_idx)%}

                weight = weights_smem[{{weight_start}} + mul_offset];

                {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
                    offset = {{ L1.slices()[u].start}} + mul_offset * {{L1[u].ir.dim}};
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
                    offset = {{ L3.slices()[w].start}} + mul_offset * {{L3[w].ir.dim}};
                    {{transpose_store(L3[w].mul, L3[w].ir.dim, 'L3_smem', 'offset', 'l3_vec', '+=', 'weight')}}
                {%- endif %}
            {%- endfor %}
        }
    }
}

__device__ __forceinline__ void backward_loop_unroll(
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

    {% set num_scratch_reg = 1 %}
    float scratch1[{{num_scratch_reg}}];
    float scratch2[{{num_scratch_reg}}];

    {%- set num_interact = interactions | length %}

    for(int mul_offset = 0; mul_offset < {{L1[0].mul}}; mul_offset += {{forward_config.warp_size}}) {
        //if(mul_offset + lane_id < {{L1[0].mul}}) {     # shfl_sync breaks with conditional 
            {%- for k in range(num_interact) %}
                {%- set u, v, w, instruction_idx, tensor = interactions[k] %}
                {%- set weight_start, _, _ = config.weight_range_and_shape_for_instruction(instruction_idx)%}
                weight = weights_smem[{{weight_start}} + mul_offset];
                weight_grad = weights_grad_smem[{{weight_start}} + mul_offset];

                {%- if k == 0 or interactions[k][0] != interactions[k-1][0] %}
                    offset = {{ L1.slices()[u].start}} + mul_offset * {{L1[u].ir.dim}};
                    {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_smem', 'offset', 'l1_vec')}}
                    {{transpose_load(L1[u].mul, L1[u].ir.dim, 'L1_grad_smem', 'offset', 'l1_grad')}}
                {%- endif %}

                {%- if k == 0 or interactions[k][1] != interactions[k-1][1] %}
                    #pragma unroll
                    for(int j = 0; j < {{L2[v].ir.dim}}; j++) {
                        l2_vec[j] = L2_smem[j + {{L2.slices()[v].start}}];
                        l2_grad[j] = L2_grad_smem[j + {{L2.slices()[v].start}}];
                    }
                {%- endif %}

                {%- if k == 0 or interactions[k][2] != interactions[k-1][2] %}
                    offset = {{ L3.slices()[w].start}} + mul_offset * {{L3[w].ir.dim}};
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
                    offset = {{ L1.slices()[u].start}} + mul_offset * {{L1[u].ir.dim}};
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
                            L2_grad_smem[j + {{L2.slices()[v].start}}] = l2_grad[j];
                    }
                    __syncwarp();
                {%- endif %}

                weights_grad_smem[mul_offset + {{weight_start}}] = weight_grad; 
            {%- endfor %}
        //}
    }
}