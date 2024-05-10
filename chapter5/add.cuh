#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef USE_DP
    typedef double real;        // 双精度
    const real EPSILON = 1.0e-15;
#else
    typedef float real;        // 单精度
    const real EPSILON = 1.0e-6;
#endif


#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char *file, int line);

// 核函数
__global__ void add_kernel(const real* x,const real* y, real* z, const int N);

// 重载设备函数
__device__ real add_in_device(const real x,const real y);
__device__ void add_in_device(const real x, const real y, real &z);

// 主机函数
void check(const real* z, const int N);