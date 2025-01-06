{%- macro generate_matmul(name, M, N, K, TILES_PER_ROW, OUTPUT_RMAJOR, A_CMAJOR=True, B_RMAJOR=True) %}

{%-set TILES_PER_COL = 32 // TILES_PER_ROW %}

__device__ __forceinline__ void {{name}}(const float* __restrict__ A, const float* __restrict__ B, float* C) {    
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = t_idx % 32;

    int const rpt = {{(M + TILES_PER_COL - 1) // TILES_PER_COL}};
    int const cpt = {{(N + TILES_PER_ROW - 1) // TILES_PER_ROW}};

    float row[cpt];
    float col[rpt];
    float tile[rpt][cpt];

    int TI_idx = lane_id / {{TILES_PER_ROW}}; 
    int TJ_idx = lane_id % {{TILES_PER_ROW}};
    int is = TI_idx * rpt; int ie = (TI_idx + 1) * rpt;
    int js = TJ_idx * cpt; int je = (TJ_idx + 1) * cpt;
    int ist = min(is, {{M}}); int iet = min(ie, {{M}});
    int jst = min(js, {{N}}); int jet = min(je, {{N}});

    // Zero the output tile
    #pragma unroll
    for(int i = 0; i < rpt; i++) {
        #pragma unroll
        for(int j = 0; j < cpt; j++) {
            tile[i][j] = 0.0f;
        }
    }

    for(int k = 0; k < {{K}}; k++) {
        #pragma unroll
        for(int i = 0; i < rpt; i++) {
            if(ist + i < {{M}}) {
                col[i] = A[k * {{M}} + ist + i];
            }
        }

        #pragma unroll
        for(int j = 0; j < cpt; j++) {
            if(jst + j < {{N}}) {
                row[j] = B[k * {{N}} + jst + j];
            }
        }

        #pragma unroll
        for(int i = 0; i < rpt; i++) {
            #pragma unroll
            for(int j = 0; j < cpt; j++) {
                if(ist + i < {{M}} && jst + j < {{N}}) {
                    tile[i][j] += col[i] * row[j];
                }
            }
        }
    }

    // Store the output
    #pragma unroll
    for(int i = 0; i < rpt; i++) {
        for(int j = 0; j < cpt; j++) {
            if(i + ist < {{M}} && j + jst < {{N}}) {
                {%- if OUTPUT_RMAJOR %}
                    C[(i + ist) * {{N}} + j + jst] += tile[i][j];
                {%- else %}
                    C[(j + jst) * {{M}} + i + ist] += tile[i][j];
                {%- endif %}
            }
        } 
    }
}

{%- endmacro %}