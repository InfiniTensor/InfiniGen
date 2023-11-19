#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

// #include "cutlass/util/print_error.hpp"
// #include "cutlass/util/GPU_Clock.hpp"

// #define CUTLASS_ENABLE_CUBLAS 1

// #if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
// #include "cutlass/util/cublas_wrappers.hpp"
// #endif

#include "test_gemm_cuda.h"

void test_cute_gemm(int m, int n, int k) {
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

  double gflops = (2.0 * m * n * k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //
  // cuBLas
  //

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Run once
  d_C = h_C;
  blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                     d_A.data().get(), m, d_B.data().get(), n, &beta,
                     d_C.data().get(), m);
  CUTE_CHECK_LAST();

  thrust::host_vector<TC> cublas_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                       d_A.data().get(), m, d_B.data().get(), n, &beta,
                       d_C.data().get(), m);
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time,
         cublas_time * 1000);

#else

  std::cout << "Verification by comparison with cuBLAS is disabled, "
               "either because the CMake option CUTLASS_ENABLE_CUBLAS "
               "was explicitly set to OFF, or because CMake could not find "
               "cuBLAS.  "
               "If you would like to enable verification with cuBLAS, "
               "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
               "rerun CMake, and recompile this example.\n";

#endif  // CUTLASS_ENABLE_CUBLAS

  //
  // CuTe
  //

  // Run once (and check)
  d_C = h_C;
  Graph_0_kernel(m, n, k, alpha, d_A.data().get(), m, d_B.data().get(), n, beta,
                 d_C.data().get(), m);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    Graph_0_kernel(m, n, k, alpha, d_A.data().get(), m, d_B.data().get(), n,
                   beta, d_C.data().get(), m);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time,
         cute_time * 1000);

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

  auto host_matrix_to_const_column_major_cute_tensor =
      [](const auto& X, int num_rows, int num_cols, int LDX) {
        const auto shape = cute::Shape<int, int>{num_rows, num_cols};
        const auto strides = cute::Stride<int, int>{1, LDX};
        return cute::make_tensor(X.data(), cute::make_layout(shape, strides));
      };

  const auto A_view =
      host_matrix_to_const_column_major_cute_tensor(h_A, m, k, m);
  // B^T is k x n, so B is n x k.
  const auto B_view =
      host_matrix_to_const_column_major_cute_tensor(h_B, n, k, n);
  const auto C_computed_view =
      host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m);
  const auto C_expected_view =
      host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m);
  print_matrix_multiply_mollified_relative_error(
      "float", A_view, B_view, C_computed_view, C_expected_view);

#endif  // CUTLASS_ENABLE_CUBLAS
}

int main() {
  int M = 5120;
  int N = 5120;
  int K = 5120;
  test_cute_gemm(M, N, K);
}