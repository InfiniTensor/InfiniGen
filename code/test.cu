
#include <cuda.h>

__device__ void ParallelTask_0(float *Data_0, float *Data_1, float *Data_3) {
  char cache[20480];
  
  ((float*)(cache))[0] = Data_0[0 + blockIdx.x * 1024 + threadIdx.x];
  ((float*)(cache))[1] = Data_1[0 + blockIdx.x * 1024 + threadIdx.x];
  ((float*)(cache))[2] = ((float*)(cache))[0] + ((float*)(cache))[1];
  ((float*)(cache))[0] = ((float*)(cache))[1] + ((float*)(cache))[2];
  Data_3[0 + blockIdx.x * 1024 + threadIdx.x] = ((float*)(cache))[0];
}

__global__ void Graph_0_kernel(float *Data_0, float *Data_1, float *Data_3) {
  ParallelTask_0(Data_0, Data_1, Data_3);
}

void Graph_0(cudaStream_t queue, float *Data_0, float *Data_1, float *Data_3) {
  int numBlocks = 4, threadsPerBlock = 1024;
  Graph_0_kernel<<<numBlocks, threadsPerBlock, 0, queue>>>(Data_0, Data_1, Data_3);
}
