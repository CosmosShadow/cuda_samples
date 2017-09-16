#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define DATA_SIZE 1024


__constant__ int num[DATA_SIZE];

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
__global__ static void sumOfSquares(int* result, clock_t* time)
{
    int sum = 0;
    clock_t start = clock();

    for (int i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    *result = sum;
    *time = clock() - start;
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


int main()
{

    //CUDA 初始化
    if (!InitCUDA()) {
        return 0;
    }

    //生成随机数
    int data[DATA_SIZE];
    GenerateNumbers(data, DATA_SIZE);

    // 把内容copy到常量内存中去
    cudaMemcpyToSymbol(num, data, sizeof(int)*DATA_SIZE);

    /*把数据复制到显卡内存中*/
    int* result;
    clock_t* time;
    cudaMalloc((void**)&result, sizeof(int));
    cudaMalloc((void**)&time, sizeof(clock_t));

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares << <1, 1, 0 >> >(result, time);

    /*把结果从显示芯片复制回主内存*/

    int sum;
    clock_t time_used;

    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);

    //Free
    cudaFree(result);
    cudaFree(time);

    printf("GPUsum: %d time: %fs\n", sum, time_used * 1.0 / cudaGetClockRate());

    sum = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i] * data[i];
    }
    printf("CPUsum: %d \n", sum);

    return 0;
}