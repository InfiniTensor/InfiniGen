#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <test_cos_cuda.h>
#include <test_sin_cuda.h>
#include <test_tanh_cuda.h>

#define LEN 100
#define EPS 1e-7

int main() {
  cudaStream_t queue;
  cudaSetDevice(0);
  cudaStreamCreate(&queue);

  float *host_dst = new float[LEN];
  float *host_src = new float[LEN];

  for (int i = 0; i < LEN; i++) {
    host_src[i] = i;
  }

  float *gpu_src;
  float *gpu_dst;
  cudaMalloc((void **)&gpu_src, LEN * sizeof(float));
  cudaMalloc((void **)&gpu_dst, LEN * sizeof(float));

  cudaMemcpy(gpu_src, host_src, LEN * sizeof(float), cudaMemcpyHostToDevice);

  Graph_0(queue, gpu_src, gpu_dst);
  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, gpu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    float res = cosf(host_src[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test cos failed.\n");
      exit(-1);
    }
  }
  printf("Test cos passed.\n");

  Graph_1(queue, gpu_src, gpu_dst);
  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, gpu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    float res = sinf(host_src[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test sin failed.\n");
      exit(-1);
    }
  }
  printf("Test sin passed.\n");

  Graph_2(queue, gpu_src, gpu_dst);
  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, gpu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    float res = tanhf(host_src[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test tanh failed.\n");
      exit(-1);
    }
  }
  printf("Test tanh passed.\n");

  printf("CUDA Unary Test PASS!\n");
  cudaStreamDestroy(queue);
  delete host_dst;
  delete host_src;
  cudaFree(gpu_src);
  cudaFree(gpu_dst);

  return 0;
}
