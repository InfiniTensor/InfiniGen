#include <cnrt.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <test_floordiv_bang.h>
#include <test_floormod_bang.h>

#define LEN 100
#define EPS 1e-5

int main() {
  cnrtQueue_t queue;
  CNRT_CHECK(cnrtSetDevice(0));
  CNRT_CHECK(cnrtQueueCreate(&queue));

  float *host_dst = new float[LEN];
  float *host_a = new float[LEN];
  float *host_b = new float[LEN];

  std::random_device seed;
  std::ranlux48 engine(seed());
  std::uniform_real_distribution<> distrib(0, 100);

  for (int i = 0; i < LEN; i++) {
    host_a[i] = distrib(engine);
    host_b[i] = distrib(engine);
  }

  float *mul_a;
  float *mul_b;
  float *mul_dst;
  cnrtMalloc((void **)&mul_a, LEN * sizeof(float));
  cnrtMalloc((void **)&mul_b, LEN * sizeof(float));
  cnrtMalloc((void **)&mul_dst, LEN * sizeof(float));

  cnrtMemcpy(mul_a, host_a, LEN * sizeof(float), cnrtMemcpyHostToDev);
  cnrtMemcpy(mul_b, host_b, LEN * sizeof(float), cnrtMemcpyHostToDev);

  Graph_0(queue, mul_a, mul_b, mul_dst);
  cnrtQueueSync(queue);

  cnrtMemcpy(host_dst, mul_dst, LEN * sizeof(float), cnrtMemcpyDevToHost);

  for (int i = 0; i < LEN; i++) {
    float res = fmod(host_a[i], host_b[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test floormod failed.\n");
      exit(-1);
    }
  }

  printf("Test floormod passed.\n");

  Graph_1(queue, mul_a, mul_b, mul_dst);
  cnrtQueueSync(queue);

  cnrtMemcpy(host_dst, mul_dst, LEN * sizeof(float), cnrtMemcpyDevToHost);

  for (int i = 0; i < LEN; i++) {
    float res = floor(host_a[i] / host_b[i]);
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
      printf("Test floordiv failed.\n");
      exit(-1);
    }
  }

  printf("Test floordiv passed.\n");
  printf("BANG BINARY TEST PASS!\n");
}
