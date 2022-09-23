/** -*- c++ -*-
 * @file
 */
#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define GPU_CHK() { \
    cudaDeviceSynchronize(); \
    cudaError_t code = cudaGetLastError(); \
    if (code != cudaSuccess) { \
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
        cudaDeviceReset(); \
    } \
}
#define K_RUN(...) { cudaLaunchCooperativeKernel(__VA_ARGS__); GPU_CHK(); }

__global__ void
k_dev(cg::thread_group b, int *tmp) {
    const int tx = threadIdx.x;
    auto b1 = cg::this_thread_block();
    if (tx == 0) {
        printf("\tk_dev: b1.size=%d, b.size=%d\n", b1.size(), b.size());
    }
    atomicAdd(tmp, tx+1);
    b1.sync();
}

__global__ void
k_sum(float *sum) {
    int *tmp = new int;
    
    const int tx = threadIdx.x, j = tx + blockIdx.x * blockDim.x;
    auto b = cg::this_thread_block();
    auto g = cg::this_grid();

    if (j == 0) {
        *sum = 0.01f * (float)b.size() + (float)g.size();
        printf("\nk_sum: b.size=%d, g.size=%ld, is_valid=%d\n",
               b.size(), g.size(), g.is_valid());
        
        k_dev<<<1, b.size()/8>>>(b, tmp);
        b.sync();
        ///__syncthreads();               /// deadlock
        printf("tmp=%d\n", *tmp);
    }
    
    if (g.is_valid()) g.sync();
    b.sync();
    
    delete tmp;
}

int main() {
    float h_sum = 0.0f, *d_sum;
    cudaMalloc(&d_sum, sizeof(float));

    k_sum<<<2,32>>>(d_sum);
    GPU_CHK();
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("<<<>>> h_sum=%f\n", h_sum);
/*
    h_sum = 0.0f;
    void *args[] = { d_sum };
    K_RUN((void*)k_sum, 2, 32, args);

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    GPU_CHK();
    printf("K_RUN h_sum=%f\n", h_sum);
*/    
    cudaFree(d_sum);
    return 0;
}
