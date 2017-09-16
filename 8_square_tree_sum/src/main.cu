#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

//1M
#define DATA_SIZE 1024*1024
#define THREAD_NUM 128
#define BLOCK_NUM 128

int data[DATA_SIZE];

//产生大量0-9之间的随机数
void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %lu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %lu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %lu.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %lu.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %lu.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
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
            printDeviceProp(prop);
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
    //声明一块共享内存
    extern __shared__ int shared[];
    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;
    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;
    shared[tid] = 0;
    int i;
    //记录运算开始的时间
    if (tid == 0) time[bid] = clock();
    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录，每个 block 都会记录开始时间及结束时间
    //thread需要同时通过tid和bid来确定，同时不要忘记保证内存连续性
    for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        shared[tid] += num[i] * num[i] * num[i];
    }
    //同步 保证每个 thread 都已经把结果写到 shared[tid] 里面
    __syncthreads();
    //使用线程0完成加和
    if (tid == 0){
        for (i = 1; i < THREAD_NUM; i++){
            shared[0] += shared[i];
        }
        result[bid] = shared[0];
    }
    //计算时间的动作，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid + BLOCK_NUM] = clock();
}


float cudaGetClockRate()
{
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess){
        return prop.clockRate * 1000.0;
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
    cudaMalloc((void**)&result, sizeof(int)*BLOCK_NUM);
    cudaMalloc((void**)&time, sizeof(clock_t)* BLOCK_NUM * 2);

    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares << < BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result, time);

    // Stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*把结果从显示芯片复制回主内存*/
    int sum[THREAD_NUM*BLOCK_NUM];
    clock_t time_use[BLOCK_NUM * 2];
    cudaMemcpy(&sum, result, sizeof(int)* BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_use, time, sizeof(clock_t)* BLOCK_NUM * 2, cudaMemcpyDeviceToHost);

    //Free
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for (int i = 0; i < BLOCK_NUM; i++) {
        final_sum += sum[i];
    }

    //采取新的计时策略 把每个 block 最早的开始时间，和最晚的结束时间相减，取得总运行时间
    clock_t max_time = time_use[BLOCK_NUM] - time_use[0];
    for (int i = 1; i < BLOCK_NUM; i++) {
        // 从下面的输出可以看到，每个block都有自己的定时器，所有不能取所有最小的开始时间、最大的结束时间
        // printf("start: %u   end: %u\n", time_use[i], time_use[i+BLOCK_NUM]);
        if (max_time < time_use[i+BLOCK_NUM] - time_use[i]){
            max_time = time_use[i+BLOCK_NUM] - time_use[i];
        }
    }
    float time_in_s = float(max_time) / cudaGetClockRate();
    printf("GPUsum: %d  gputime: %f\n", final_sum, time_in_s);


    final_sum = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        final_sum += data[i] * data[i] * data[i];
    }
    printf("CPUsum: %d \n", final_sum);

    printMemoryBandwidth(DATA_SIZE*sizeof(int)/8, time_in_s);

    printf("全局内存带宽\n");
    printMemoryBandwidth(DATA_SIZE*sizeof(int)/8, elapsedTime);

    return 0;
}