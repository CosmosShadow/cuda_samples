#include "cuda_runtime.h"  
#include <iostream>
#include <stdio.h>  
#include <math.h>  

#define N (1024*1024)  
#define FULL_DATA_SIZE N*20  

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}

bool isSupportOverlap()
{
    //获取设备属性
    cudaDeviceProp prop;
    int deviceID;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);

    //检查设备是否支持重叠功能
    if (!prop.deviceOverlap){
        printf("No device will handle overlaps. so no speed up from stream.\n");
        return false;
    } else {
        return true;
    }
}

__global__ void kernel(int* a, int *b, int*c)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)
    {
        c[threadID] = (a[threadID] + b[threadID]) / 2;
    }
}

int main()
{
    if (!InitCUDA() | !isSupportOverlap()) {
        return 0;
    }

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    //在GPU上分配内存
    cudaMalloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, FULL_DATA_SIZE * sizeof(int));

    //在CPU上分配可分页内存
    host_a = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
    host_b = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
    host_c = (int*)malloc(FULL_DATA_SIZE * sizeof(int));

    //主机上的内存赋值
    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = i;
        host_b[i] = FULL_DATA_SIZE - i;
    }

    //启动计时器
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //从主机到设备复制数据
    cudaMemcpy(dev_a, host_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    kernel << <FULL_DATA_SIZE / 1024, 1024 >> > (dev_a, dev_b, dev_c);

    //数据拷贝回主机
    cudaMemcpy(host_c, dev_c, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //计时结束
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "消耗时间： " << elapsedTime << std::endl;

    //输出前10个结果
    for (int i = 0; i < 10; i++)
    {
        std::cout << host_c[i] << std::endl;
    }

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}