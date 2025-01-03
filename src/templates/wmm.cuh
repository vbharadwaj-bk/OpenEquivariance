__global__ void warp_matmul(float* A, float* B, float* C) {    
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(t_idx == 0) {
        for(int i = 0; i < {{M}}; i++) {
            for(int j = 0; j < {{N}}; j++) {
                float sum = 0;
                for(int k = 0; k < {{K}}; k++) {
                    sum += A[i * {{K}} + k] * B[k * {{N}} + j];
                }
                C[i * {{N}} + j] = sum;
            }
        }
    }
}