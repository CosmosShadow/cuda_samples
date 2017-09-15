#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define DATA_SIZE 1024*1024
#define THREAD_NUM 256

int data[DATA_SIZE];

//产生大量0-9之间的随机数
void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}

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


// __global__ 函数 (GPU上执行) 计算立方和
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x; 

    int sum = 0;
    int i;

    //记录运算开始的时间
    clock_t start;

    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录
    if (tid == 0) start = clock();

    // 显存存取优化: 
    // thread1 -> thread2 -> thread3 -> ...
    // 这样让显存连续读取
    for (i = tid; i < DATA_SIZE; i += THREAD_NUM) {
        sum += num[i] * num[i] * num[i];
    }
    result[tid] = sum;
    //计算时间的动作，只在 thread 0（即 threadIdx.x = 0 的时候）进行
    if (tid == 0) *time = clock() - start;

}


int cudaGetClockRate()
{
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess){
        return prop.clockRate * 1000;
    } else {
        std::cout << "cudaGetClockRate fails" << std::endl;
        return 10^9;
    }
}

// 打印内存带宽
void printMemoryBandwidth(float mem_size, float time_cost)
{
    float size_M_per_s = (mem_size / time_cost)  / float(1024*1024);
    printf("内存带宽 %.2f M/s\n", size_M_per_s);
}


int main()
{

    //CUDA 初始化
    if (!InitCUDA()) {
        return 0;
    }

    //生成随机数
    GenerateNumbers(data, DATA_SIZE);

    /*把数据复制到显卡内存中*/
    int* gpudata, *result;
    clock_t* time;

    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果，time用来存储运行时间 )
    cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
    cudaMalloc((void**)&time, sizeof(clock_t));
    //扩大记录结果的内存，记录THREAD_NUM个结果
    cudaMalloc((void**)&result, sizeof(int) * THREAD_NUM);

    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares << <1, THREAD_NUM, 0 >> >(gpudata, result, time);


    /*把结果从显示芯片复制回主内存*/
    int sum[THREAD_NUM];
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);

    //Free
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for (int i = 0; i < THREAD_NUM; i++) {
        final_sum += sum[i];
    }
    float time_in_s = time_used * 1.0 / cudaGetClockRate();
    printf("GPUsum: %d time: %fs\n", final_sum, time_in_s);

    final_sum = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        final_sum += data[i] * data[i] * data[i];
    }
    printf("CPU sum: %d \n", final_sum);

    printMemoryBandwidth(DATA_SIZE*sizeof(int)/8, time_in_s);

    return 0;
}