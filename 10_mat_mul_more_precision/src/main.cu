#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

//1M
#define THREAD_NUM 256
#define MATRIX_SIZE 1000
#define blocks_num ((MATRIX_SIZE*MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM)

//生成随机矩阵
void getMat(float* mat, int n) 
{
    int i, j; 
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat[i * n + j] = (float)rand() / (float)RAND_MAX + (float)rand() / ((float)RAND_MAX * (float)RAND_MAX);
        }
    }
}

void printMat(float* mat, int n)
{
    printf("-------------------------------------------------------\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", mat[i*n + j]);
        }
        printf("\n");
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


__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n, clock_t* time)
{
    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    //一块block有多少个thread
    const int bdim = blockDim.x;

    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
    const int idx = bid * bdim + tid;
    if (idx < n * n){
        const int row = idx / n;
        const int column = idx % n;

        //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录，每个 block 都会记录开始时间及结束时间
        if (tid == 0) time[bid] = clock();

        // Kahan’s Summation Formula: 记录损失的部分，提高计算精度
        float t = 0;
        float y = 0;
        for (int i = 0; i < n; i++){
            float r;
            y -= a[row * n + i] * b[i * n + column];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
        c[row * n + column] = t;

        //计算时间,记录结果，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
        if (tid == 0) time[bid + blocks_num] = clock();
    }
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

    //定义矩阵
    float *a, *b, *c, *d;
    int n = MATRIX_SIZE;
    //分配内存
    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n); 
    d = (float*)malloc(sizeof(float)* n * n);

    //生成随机矩阵
    getMat(a, n);
    getMat(b, n);

    // 分配显存
    float *cuda_a, *cuda_b, *cuda_c;
    clock_t* time;
    //cudaMalloc 取得一块显卡内存 
    cudaMalloc((void**)&cuda_a, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float)* n * n);
    cudaMalloc((void**)&time, sizeof(clock_t)* blocks_num * 2);

    /*把数据复制到显卡内存中*/
    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(cuda_a, a, sizeof(float)* n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float)* n * n, cudaMemcpyHostToDevice);

    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    matMultCUDA << < blocks_num, THREAD_NUM, 0>> >(cuda_a , cuda_b , cuda_c , n , time);

    // Stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*把结果从显示芯片复制回主内存*/
    clock_t time_use[blocks_num * 2];
    cudaMemcpy(c, cuda_c, sizeof(float)* n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_use, time, sizeof(clock_t)* blocks_num * 2, cudaMemcpyDeviceToHost);

    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    cudaFree(time);

    //采取新的计时策略 把每个 block 最早的开始时间，和最晚的结束时间相减，取得总运行时间
    clock_t max_time = time_use[blocks_num] - time_use[0];
    for (int i = 1; i < blocks_num; i++) {
        // 从下面的输出可以看到，每个block都有自己的定时器，所有不能取所有最小的开始时间、最大的结束时间
        // printf("start: %u   end: %u\n", time_use[i], time_use[i+blocks_num]);
        if (max_time < time_use[i+blocks_num] - time_use[i]){
            max_time = time_use[i+blocks_num] - time_use[i];
        }
    }
    float time_in_s = float(max_time) / cudaGetClockRate();
    printf("gputime: %f\n", time_in_s);

    printMemoryBandwidth(MATRIX_SIZE*MATRIX_SIZE*sizeof(int)/8, time_in_s);
    printf("全局内存带宽\n");
    printMemoryBandwidth(MATRIX_SIZE*MATRIX_SIZE*sizeof(int)/8, elapsedTime);

    //CPU矩阵乘法，存入矩阵d
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){ 
            double t = 0;
            for (int k = 0; k < n; k++){ 
                t += a[i * n + k] * b[k * n + j]; 
            }
            d[i * n + j] = t; 
        } 
    }

    //验证正确性与精确性
    float max_err = 0;
    float average_err = 0; 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (d[i * n + j] != 0){ 
                //fabs求浮点数x的绝对值
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);
                // float err = fabs(c[i * n + j] - d[i * n + j]);
                if (max_err < err) max_err = err; 
                average_err += err; 
            } 
        } 
    }
    printf("Max error: %g Average error: %g\n",max_err, average_err / (n * n));

    // printMat(a, n);
    // printMat(b, n);
    // printMat(c, n);
    // printMat(d, n);

    return 0;
}