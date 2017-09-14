# Append current NVCC flags by something, eg comput capability
SET(CUDASRCS main.cu)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
CUDA_ADD_EXECUTABLE(exe ${CUDASRCS})