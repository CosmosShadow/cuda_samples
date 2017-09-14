#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

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

bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        std::cout << "no cuda device found." << std::endl;
        return false;
    }

    int i, index;

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            printDeviceProp(prop);
            if (prop.major >= 1) {
                index = i;
            }
        }
    }

    cudaSetDevice(index);
    std::cout << "cuda initialized with set device " << index << std::endl;
    return true;
}

int main() 
{
    if (!InitCUDA()) { 
        return 0; 
    }
    std::cout << "cuda initialized." << std::endl;
    return 0;
}







