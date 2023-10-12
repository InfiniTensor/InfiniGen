#include <cuda_runtime.h>
#include <test.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define EPS 1e-7
#define LEN 2050

int main(void)
{
  cudaStream_t queue;
  cudaSetDevice(0);
  cudaStreamCreate(&queue);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float* host_dst = (float*)malloc(LEN * sizeof(float));
  float* host_src1 = (float*)malloc(LEN * sizeof(float));
  float* host_src2 = (float*)malloc(LEN * sizeof(float));

  for (int i = 0; i < LEN; i++) {
    host_src1[i] = i;
    host_src2[i] = i;
  }

  float* mlu_dst;
  float* mlu_src1;
  float* mlu_src2;
  cudaMalloc((void**)&mlu_dst, LEN * sizeof(float));
  cudaMalloc((void**)&mlu_src1, LEN * sizeof(float));
  cudaMalloc((void**)&mlu_src2, LEN * sizeof(float));

  cudaMemcpy(mlu_src1, host_src1, LEN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mlu_src2, host_src2, LEN * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, queue);
  Graph_0(queue, mlu_src1, mlu_src2, mlu_dst);
  cudaEventRecord(end, queue);

  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, mlu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    if (fabsf(host_dst[i] - i) > EPS) {
      printf("%f expected, but %f got!\n", (float)(i), host_dst[i]);
    }
  }

  float timeTotal;
  cudaEventElapsedTime(&timeTotal, start, end);
  printf("Total Time: %.3f ms\n", timeTotal / 1000.0);

  cudaStreamDestroy(queue);

  cudaFree(mlu_dst);
  cudaFree(mlu_src1);
  cudaFree(mlu_src2);
  free(host_dst);
  free(host_src1);
  free(host_src2);

  return 0;
}
