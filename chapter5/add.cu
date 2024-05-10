#include "add.cuh"

const real c=3.57;

bool __check_cuda_runtime(cudaError_t code, const char* op, const char *file, int line)
{
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

__global__ void add_kernel(const real* x,const real* y, real* z, const int N)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > N) return;

    if(idx % 5 == 0)
    {
        z[idx] = add_in_device(x[idx], y[idx]);
    }
    else
    {
        add_in_device(x[idx], y[idx], z[idx]);
    }
}

__device__ real add_in_device(const real x,const real y)
{
    return x + y;
}

__device__ void add_in_device(const real x, const real y, real &z)
{
    z = x + y;
}

void check(const real* z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N ;++i)
    {
        if (fabs(z[i] - c) > EPSILON)
        {
            has_error = true;
        }
    }

    printf("cuda; %s\n", has_error ? "has error" : "no error");
}