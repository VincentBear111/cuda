#include <cstdio>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <math.h>
#include <stdio.h>

const double EPSLION = 1.0e-10;
const double a = 1.23l;
const double b = 2.34;
const double c = 3.57;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) 
{
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

// 核函数
__global__ void add_kernel(const double *x, const double *y, double *z, const int N);

// 重载设备函数
__device__ double add_in_device(const double x, const double y);
__device__ void add_in_device(const double x, const double y, double& z);

// 主机函数
void check(const double *z, const int N);


int main()
{
    const int N = 1e4;
    const int byteCount = sizeof(double) * N;

    // 主机申请内存
    // 支持使用 new-delete 方式创建和释放内存
    double* x_host = (double*)malloc(byteCount);
    double* y_host = (double*)malloc(byteCount);
    double* z_host = (double*)malloc(byteCount);

    // 初始化主机数据
    for (int i = 0; i < N; i++)
    {
        x_host[i] = a;
        y_host[i] = b;
    }

    // 申请设备内存
    double* x_device;
    double* y_device;
    double* z_device;
    checkRuntime(cudaMalloc((void**)&x_device, byteCount));
    cudaMalloc((void**)&y_device, byteCount);
    cudaMalloc((void**)&z_device, byteCount);

    // 复制数据到设备
    cudaMemcpy(x_device, x_host, byteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, byteCount, cudaMemcpyHostToDevice);

    // 在设备中执行计算
    const int block_size = 128;     // 线程数应该不少于计算数目
    const int grid_size = N / 128 + 1;
    add_kernel<<<grid_size, block_size>>>(x_device, y_device, z_device, N);

    // 复制数据到主机
    cudaMemcpy(z_host, z_device, byteCount, cudaMemcpyDeviceToHost);
    // 主机和设备之间的数据拷贝会隐式的同步主机和设备，一般要获得精确地出错位置，
    // 还是需要显示的同步。例如调用 'cudaDeviceSynchronize()'
    checkRuntime(cudaDeviceSynchronize());

    // 检查结果
    check(z_host, N);

    // 释放主机和设备内存
    free(x_host);
    free(y_host);
    free(z_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(z_host);
    

    return 0;
}


__global__ void add_kernel(const double *x, const double *y, double *z, const int N)
{
    /*
    *   在主机设备中需要依次对每个元素进行操作，需要使用一个循环。
    *   在设备函数中，因为采用“单指令-多线程”方式，所以可以去掉循环，只需要将数组元素和线程索引一一对应即可。
    */

   const int n = blockDim.x * blockIdx.x + threadIdx.x;
   if (n > N)
   {
        return;
   }
   
   if (n % 5 == 0)
   {
        z[n] = add_in_device(x[n], y[n]);
   }
   else
   {
        add_in_device(x[n], y[n], z[n]);
   }

}

__device__ double add_in_device(const double x, const double y)
{
    return x + y;
}

__device__ void add_in_device(const double x, const double y, double &z)
{
    z = x + y;
}

void check(const double *z, const int N)
{
    bool has_error = false;

    for (int i = 0; i < N; i++)
    {
        if (fabs(z[i] - c) > EPSLION)
        {
            has_error = true;
        }
        
    }
    
    printf("cuda; %s\n", has_error ? "has error" : "no error");
}