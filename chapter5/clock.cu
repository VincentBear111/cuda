#include "add.cuh"
#include "clock.cuh"


const real a = 1.23;
const real b = 2.34;

void cuda_clock()
{
    const int N = 1e6;
    const int byteCount = sizeof(real) * N;

    // cuda 计时
    float elapsed_time = 0;
    float cur_time = 0;
    cudaEvent_t start, stop;
    checkRuntime(cudaEventCreate(&start));      // 创建 cuda 事件对象
    checkRuntime(cudaEventCreate(&stop));
    checkRuntime(cudaEventRecord(start));       // 开始计时
    checkRuntime(cudaEventQuery(start));        // 强制刷新 cuda 执行流

    // ----------------------------------------
    real* x_host, * y_host, * z_host;
    x_host = new real[byteCount];
    y_host = new real[byteCount];
    z_host = new real[byteCount];
    if(x_host==NULL || y_host==NULL || z_host==NULL)\
    {
        printf("Error: host memory malloc failed.\n");
        return;
    }

    for(int i = 0; i < N; i++)
    {
        x_host[i]=a;
        y_host[i]=b;
    }

    // 主机申请及初始化内存耗时
    checkRuntime(cudaEventRecord(stop));
    checkRuntime(cudaEventSynchronize(stop));       // 强制同步，让主机等待cuda事件执行完毕
    checkRuntime(cudaEventElapsedTime(&cur_time, start, stop));
    printf("host memeory malloc and copy : %f ms.\n", cur_time - elapsed_time);
    elapsed_time = cur_time;

    // ----------------------------------------
    real *x_device, *y_device, *z_device;
    checkRuntime(cudaMalloc(&x_device, byteCount));
    checkRuntime(cudaMalloc(&y_device, byteCount));
    checkRuntime(cudaMalloc(&z_device, byteCount));
    checkRuntime(cudaMemcpy(x_device, x_host, byteCount, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(y_device, y_host, byteCount, cudaMemcpyHostToDevice));
    // -----------------------------------------

    // 设备内存申请和拷贝耗时
    checkRuntime(cudaEventRecord(stop));
    checkRuntime(cudaEventSynchronize(stop));       // 强制同步，让主机等待cuda事件执行完毕
    checkRuntime(cudaEventElapsedTime(&cur_time, start, stop));
    printf("device memeory malloc and copy : %f ms.\n", cur_time - elapsed_time);
    elapsed_time = cur_time;

    // ----------------------------------------
    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(x_device, y_device, z_device, N);
    // ----------------------------------------

    // 核函数运行耗时
    checkRuntime(cudaEventRecord(stop));
    checkRuntime(cudaEventSynchronize(stop));       // 强制同步，让主机等待cuda事件执行完毕
    checkRuntime(cudaEventElapsedTime(&cur_time, start, stop));
    printf("kernel function run time : %f ms.\n", cur_time - elapsed_time);
    elapsed_time = cur_time;

    // ----------------------------------------
    checkRuntime(cudaGetLastError());
    checkRuntime(cudaMemcpy(z_host, z_device, byteCount, cudaMemcpyDeviceToHost));
    check(z_host, N);

    // 数据拷贝耗时
    checkRuntime(cudaEventRecord(stop));
    checkRuntime(cudaEventSynchronize(stop));       // 强制同步，让主机等待cuda事件执行完毕
    checkRuntime(cudaEventElapsedTime(&cur_time, start, stop));
    printf("copy from device to host : %f ms.\n", cur_time - elapsed_time);
    elapsed_time = cur_time;

    // 内存释放
    if(x_host!=NULL)
    {   
        delete[] x_host;
    }
    if(y_host!=NULL)
    {   
        delete[] y_host;
    }
    if(z_host!=NULL)
    {   
        delete[] z_host;
    }

    checkRuntime(cudaFree(x_device));
    checkRuntime(cudaFree(y_device));
    checkRuntime(cudaFree(z_device));

}