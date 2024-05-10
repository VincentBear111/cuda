#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <stdio.h>

__global__ void hello_from_gpu()
{
    // 核函数不支持c++的iostream，只能使用printf()输出

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    printf("gpu: hello world! block(%d, %d, %d) -- thread(%d, %d, %d) \n", bx, by, bz, tz, tx, ty);
}

int main()
{
    printf("nvcc: hello world \n");

    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}