#include <cuda_runtime.h>
#include <test.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define EPS 1e-7
#define LEN 224 * 768

int main(void) {
  cudaStream_t queue;
  cudaSetDevice(0);
  cudaStreamCreate(&queue);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float *host_dst = (float *)malloc(LEN * sizeof(float));
  float *host_src1 = (float *)malloc(LEN * sizeof(float));
  float *host_src2 = (float *)malloc(LEN * sizeof(float));
  float *host_src3 = (float *)malloc(LEN * sizeof(float));
  float *host_src4 = (float *)malloc(LEN * sizeof(float));

  for (int i = 0; i < LEN; i++) {
    host_src1[i] = i;
    host_src2[i] = i;
    host_src3[i] = 1;
    host_src4[i] = i;
  }

  float *mlu_dst;
  float *mlu_src1;
  float *mlu_src2;
  float *mlu_src3;
  float *mlu_src4;
  cudaMalloc((void **)&mlu_dst, LEN * sizeof(float));
  cudaMalloc((void **)&mlu_src1, LEN * sizeof(float));
  cudaMalloc((void **)&mlu_src2, LEN * sizeof(float));
  cudaMalloc((void **)&mlu_src3, LEN * sizeof(float));
  cudaMalloc((void **)&mlu_src4, LEN * sizeof(float));

  cudaMemcpy(mlu_src1, host_src1, LEN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mlu_src2, host_src2, LEN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mlu_src3, host_src3, LEN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(mlu_src4, host_src4, LEN * sizeof(float), cudaMemcpyHostToDevice);

  int warmupRounds = 100;
  int timingRounds = 200;

  for (int i = 0; i < warmupRounds; i++) {
    Graph_0(queue, mlu_src1, mlu_src2, mlu_src3, mlu_src4, mlu_dst);
  }

  cudaEventRecord(start, queue);
  for (int i = 0; i < timingRounds; i++) {
    Graph_0(queue, mlu_src1, mlu_src2, mlu_src3, mlu_src4, mlu_dst);
  }
  cudaEventRecord(end, queue);

  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, mlu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    float res = 1 / (1 + exp(-sqrt(i + i - 1) * i));
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
    }
  }

  float timeTotal;
  cudaEventElapsedTime(&timeTotal, start, end);
  printf("Total Time: %.6f ms\n", timeTotal / timingRounds);

  cudaStreamDestroy(queue);

  cudaFree(mlu_dst);
  cudaFree(mlu_src1);
  cudaFree(mlu_src2);
  cudaFree(mlu_src3);
  cudaFree(mlu_src4);
  free(host_dst);
  free(host_src1);
  free(host_src2);
  free(host_src3);
  free(host_src4);

  return 0;
}
