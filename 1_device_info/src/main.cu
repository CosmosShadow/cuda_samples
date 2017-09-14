#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

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
            std::cout << "device " << i << " version: " << prop.major << "." << prop.minor << std::endl;
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







