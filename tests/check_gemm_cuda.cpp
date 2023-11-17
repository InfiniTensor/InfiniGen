#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_gemm_cuda.h"

void test_cute_gemm(int m, int n, int k) {
  // cute::device_init(0);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);

  for (int j = 0; j < m * k; ++j)
    h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n * k; ++j)
    h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  TI alpha = 1.0;
  TI beta = 0.0;

  d_C = h_C;
  Graph_0_kernel(m, n, k, alpha, d_A.data().get(), m, d_B.data().get(), n, beta,
                 d_C.data().get(), m);
}

int main() {
  std::cout << "Hello, World!" << std::endl;
  test_cute_gemm(256, 256, 256);
}