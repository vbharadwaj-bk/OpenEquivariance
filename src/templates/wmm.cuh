__global__ void warp_matmul(int M, int N, int K,
        void* A, void* B, void* C) {
    printf("Hello from warp_matmul\n");
}