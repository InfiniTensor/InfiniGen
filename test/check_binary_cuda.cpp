#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <test_floordiv_cuda.h>
#include <test_floormod_cuda.h>

#define LEN 100
#define EPS 1e-7

int main() {
  cudaStream_t queue;
  cudaSetDevice(0);
  cudaStreamCreate(&queue);

  std::random_device seed;
  std::ranlux48 engine(seed());
  std::uniform_int_distribution<> distrib(0, 100);

  float *host_dst = new float[LEN];
  float *host_a = new float[LEN];
  float *host_b = new float[LEN];

  for (int i = 0; i < 100; i++) {
    host_a[i] = distrib(engine);
    host_b[i] = distrib(engine);
  }

  float *gpu_a, *gpu_b;
  float *gpu_dst;
  cudaMalloc((void **)&gpu_a, LEN * sizeof(float));
  cudaMalloc((void **)&gpu_b, LEN * sizeof(float));
  cudaMalloc((void **)&gpu_dst, LEN * sizeof(float));

  cudaMemcpy(gpu_a, host_a, LEN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b, host_b, LEN * sizeof(float), cudaMemcpyHostToDevice);

  Graph_0(queue, gpu_a, gpu_b, gpu_dst);
  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, gpu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    float res = fmod(host_a[i], host_b[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test floormod failed.\n");
      exit(-1);
    }
  }
  printf("Test floormod passed.\n");

  Graph_1(queue, gpu_a, gpu_b, gpu_dst);
  cudaStreamSynchronize(queue);
  cudaMemcpy(host_dst, gpu_dst, LEN * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < LEN; i++) {
    float res = floor(host_a[i] / host_b[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test floordiv failed.\n");
      exit(-1);
    }
  }
  printf("Test floordiv passed.\n");

  printf("CUDA Binary Test PASS!\n");
  cudaStreamDestroy(queue);
  delete host_a;
  delete host_b;
  delete host_dst;
  cudaFree(gpu_a);
  cudaFree(gpu_b);
  cudaFree(gpu_dst);

  return 0;
}
