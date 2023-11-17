#include <cnrt.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <test_cos_bang.h>
#include <test_sin_bang.h>
#include <test_tanh_bang.h>

#define LEN 100
#define EPS 1e-5

int main() {
  cnrtQueue_t queue;
  CNRT_CHECK(cnrtSetDevice(0));
  CNRT_CHECK(cnrtQueueCreate(&queue));

  float *host_dst = new float[LEN];
  float *host_src = new float[LEN];

  std::random_device seed;
  std::ranlux48 engine(seed());
  std::uniform_real_distribution<> distrib(0, 1);

  for (int i = 0; i < LEN; i++) {
    host_src[i] = distrib(engine);
  }

  float *mul_src;
  float *mul_dst;
  cnrtMalloc((void **)&mul_src, LEN * sizeof(float));
  cnrtMalloc((void **)&mul_dst, LEN * sizeof(float));

  cnrtMemcpy(mul_src, host_src, LEN * sizeof(float), cnrtMemcpyHostToDev);

  Graph_0(queue, mul_src, mul_dst);
  cnrtQueueSync(queue);

  cnrtMemcpy(host_dst, mul_dst, LEN * sizeof(float), cnrtMemcpyDevToHost);

  for (int i = 0; i < LEN; i++) {
    float res = cosf(host_src[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test cos failed.\n");
      exit(-1);
    }
  }
  printf("Test cos passed.\n");

  Graph_1(queue, mul_src, mul_dst);
  cnrtQueueSync(queue);

  cnrtMemcpy(host_dst, mul_dst, LEN * sizeof(float), cnrtMemcpyDevToHost);

  for (int i = 0; i < LEN; i++) {
    float res = sinf(host_src[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test sin failed.\n");
      exit(-1);
    }
  }
  printf("Test sin passed.\n");

  Graph_2(queue, mul_src, mul_dst);
  cnrtQueueSync(queue);

  cnrtMemcpy(host_dst, mul_dst, LEN * sizeof(float), cnrtMemcpyDevToHost);

  for (int i = 0; i < LEN; i++) {
    float res = tanhf(host_src[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test tanh failed.\n");
      exit(-1);
    }
  }

  printf("Test tanh passed.\n");
  printf("BANG UNARY TEST PASS!\n");
}
