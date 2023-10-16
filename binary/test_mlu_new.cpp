#include <cnrt.h>
#include <test.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>

#define EPS 1e-7
#define LEN 224 * 768

int main(void) {
  cnrtQueue_t queue;
  CNRT_CHECK(cnrtSetDevice(0));
  CNRT_CHECK(cnrtQueueCreate(&queue));

  cnrtNotifier_t start, end;
  CNRT_CHECK(cnrtNotifierCreate(&start));
  CNRT_CHECK(cnrtNotifierCreate(&end));

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
  CNRT_CHECK(cnrtMalloc((void **)&mlu_dst, LEN * sizeof(float)));
  CNRT_CHECK(cnrtMalloc((void **)&mlu_src1, LEN * sizeof(float)));
  CNRT_CHECK(cnrtMalloc((void **)&mlu_src2, LEN * sizeof(float)));
  CNRT_CHECK(cnrtMalloc((void **)&mlu_src3, LEN * sizeof(float)));
  CNRT_CHECK(cnrtMalloc((void **)&mlu_src4, LEN * sizeof(float)));

  CNRT_CHECK(cnrtMemcpy(mlu_src1, host_src1, LEN * sizeof(float),
                        cnrtMemcpyHostToDev));
  CNRT_CHECK(cnrtMemcpy(mlu_src2, host_src2, LEN * sizeof(float),
                        cnrtMemcpyHostToDev));
  CNRT_CHECK(cnrtMemcpy(mlu_src3, host_src3, LEN * sizeof(float),
                        cnrtMemcpyHostToDev));
  CNRT_CHECK(cnrtMemcpy(mlu_src4, host_src4, LEN * sizeof(float),
                        cnrtMemcpyHostToDev));

  int warmupRounds = 100;
  int timingRounds = 200;

  for (int i = 0; i < warmupRounds; i++) {
    Graph_0(queue, mlu_src1, mlu_src2, mlu_src3, mlu_src4, mlu_dst);
  }
  CNRT_CHECK(cnrtPlaceNotifier(start, queue));
  for (int i = 0; i < timingRounds; i++) {
    Graph_0(queue, mlu_src1, mlu_src2, mlu_src3, mlu_src4, mlu_dst);
  }
  CNRT_CHECK(cnrtPlaceNotifier(end, queue));

  cnrtQueueSync(queue);
  CNRT_CHECK(
      cnrtMemcpy(host_dst, mlu_dst, LEN * sizeof(float), cnrtMemcpyDevToHost));

  for (int i = 0; i < LEN; i++) {
    float res = 1 / (1 + exp(-sqrt(i + i - 1) * i));
    if (fabsf(host_dst[i] - res) > EPS) {
      printf("%f expected, but %f got!\n", res, host_dst[i]);
    }
  }

  float timeTotal;
  CNRT_CHECK(cnrtNotifierDuration(start, end, &timeTotal));
  printf("Total Time: %.6f ms\n", timeTotal / timingRounds);

  CNRT_CHECK(cnrtQueueDestroy(queue));

  cnrtFree(mlu_dst);
  cnrtFree(mlu_src1);
  cnrtFree(mlu_src2);
  cnrtFree(mlu_src3);
  cnrtFree(mlu_src4);
  free(host_dst);
  free(host_src1);
  free(host_src2);
  free(host_src3);
  free(host_src4);

  return 0;
}

