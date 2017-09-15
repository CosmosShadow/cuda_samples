#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

//1M
#define DATA_SIZE 1024*1024
#define THREAD_NUM 1024
#define BLOCK_NUM 1

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
    cudaSetDevice(0);
    return true;
}


// __global__ 函数 (GPU上执行) 计算立方和
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{

    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;
    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;
    int sum = 0;
    int i;
    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid] = clock();
    //thread需要同时通过tid和bid来确定，同时不要忘记保证内存连续性
    for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        sum += num[i] * num[i] * num[i];
    }
    //Result的数量随之增加
    result[bid * THREAD_NUM + tid] = sum;
    //计算时间的动作，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid + BLOCK_NUM] = clock();

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
    cudaMalloc((void**)&result, sizeof(int)*THREAD_NUM* BLOCK_NUM);
    cudaMalloc((void**)&time, sizeof(clock_t)* BLOCK_NUM * 2);

    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares << < BLOCK_NUM, THREAD_NUM, 0 >> >(gpudata, result, time);


    /*把结果从显示芯片复制回主内存*/
    int sum[THREAD_NUM*BLOCK_NUM];
    clock_t time_use[BLOCK_NUM * 2];
    cudaMemcpy(&sum, result, sizeof(int)* THREAD_NUM*BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_use, time, sizeof(clock_t)* BLOCK_NUM * 2, cudaMemcpyDeviceToHost);

    //Free
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for (int i = 0; i < THREAD_NUM*BLOCK_NUM; i++) {
        final_sum += sum[i];
    }

    //采取新的计时策略 把每个 block 最早的开始时间，和最晚的结束时间相减，取得总运行时间
    clock_t min_start, max_end;
    min_start = time_use[0];
    max_end = time_use[BLOCK_NUM];
    for (int i = 1; i < BLOCK_NUM; i++) {
        if (min_start > time_use[i])
            min_start = time_use[i];
        if (max_end < time_use[i + BLOCK_NUM])
            max_end = time_use[i + BLOCK_NUM];
    }
    float time_in_s = (max_end - min_start) * 1.0 / cudaGetClockRate();
    printf("GPUsum: %d  gputime: %f\n", final_sum, time_in_s);


    final_sum = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        final_sum += data[i] * data[i] * data[i];
    }
    printf("CPUsum: %d \n", final_sum);

    printMemoryBandwidth(DATA_SIZE*sizeof(int)/8, time_in_s);

    return 0;
}