#include "graphs/gemm_graph.h"

namespace infini {
const std::string GemmKernel = R"(
template <class MShape, class NShape, class KShape, class TA, class AStride,
          class ABlockLayout, class AThreadLayout, class TB, class BStride,
          class BBlockLayout, class BThreadLayout, class TC, class CStride,
          class CBlockLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    CThreadLayout{}))::value) void gemm_device(MShape M, NShape N, KShape K,
                                               TA const* A, AStride dA,
                                               ABlockLayout blockA,
                                               AThreadLayout tA, TB const* B,
                                               BStride dB, BBlockLayout blockB,
                                               BThreadLayout tB, TC* C,
                                               CStride dC, CBlockLayout,
                                               CThreadLayout tC, Alpha alpha,
                                               Beta beta) {
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

    CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
    CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

    // CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));      // BLK_M
    // CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));      // BLK_N
    CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB));  // BLK_K

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ABlockLayout>];
    __shared__ TB smemB[cosize_v<BBlockLayout>];
    auto sA = make_tensor(make_smem_ptr(smemA), blockA);  // (BLK_M,BLK_K)
    auto sB = make_tensor(make_smem_ptr(smemB), blockB);  // (BLK_N,BLK_K)

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);  // (M,K)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);  // (N,K)
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);  // (M,N)

    // Get the appropriate blocks for this thread block --
    // potential for thread block locality
    auto blk_shape = make_shape(size<0>(sA), size<0>(sB),
                                size<1>(sB));  // (BLK_M,BLK_N,BLK_K)
    auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);  // (m,n,k)

    auto gA = local_tile(mA, blk_shape, blk_coord,
                         Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blk_shape, blk_coord,
                         Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blk_shape, blk_coord,
                         Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    //
    // Partition the copying of A and B tiles across the threads
    //

    // TUTORIAL: Example of simple partitioning of A|B tiles over tA|tB
    //   Default is a raked partition, but can be changed with Step<X,Y>
    //   parameter

    auto tAgA = local_partition(gA, tA, threadIdx.x);  // (THR_M,THR_K,k)
    auto tAsA = local_partition(sA, tA, threadIdx.x);  // (THR_M,THR_K)

    auto tBgB = local_partition(gB, tB, threadIdx.x);  // (THR_N,THR_K,k)
    auto tBsB = local_partition(sB, tB, threadIdx.x);  // (THR_N,THR_K)

    //
    // Define C accumulators and A/B partitioning
    //

    // TUTORIAL: Example of partitioning via projections of tC

    // Partition sA (M,K) by the rows of tC
    auto tCsA =
        local_partition(sA, tC, threadIdx.x, Step<_1, X>{});  // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    auto tCsB =
        local_partition(sB, tC, threadIdx.x, Step<X, _1>{});  // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    auto tCgC =
        local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});  // (THR_M,THR_N)

    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC);  // (THR_M,THR_N)

    // Clear the accumulators
    clear(tCrC);

#if 0
  if(thread0()) {
    print("mA\n");
    print(mA.shape()); print("\n"); print(mA.stride());
    print("\n\ngA\n");
    print(gA.shape()); print("\n"); print(gA.stride());
    print("\n\ntAgA\n");
    print(tAgA.shape()); print("\n"); print(tAgA.stride());
    print("\n\nsA\n");
    print(sA.shape()); print("\n"); print(sA.stride());
    print("\n\ntAsA\n");
    print(tAsA.shape()); print("\n"); print(tAsA.stride());
    print("\n\n");
  }
#endif

#if 0
  if(thread0()) {
    print("mB\n");
    print(mB.shape()); print("\n"); print(mB.stride());
    print("\n\ngB\n");
    print(gB.shape()); print("\n"); print(gB.stride());
    print("\n\ntBgB\n");
    print(tBgB.shape()); print("\n"); print(tBgB.stride());
    print("\n\nsB\n");
    print(sB.shape()); print("\n"); print(sB.stride());
    print("\n\ntBsB\n");
    print(tBsB.shape()); print("\n"); print(tBsB.stride());
    print("\n\n");
  }
#endif

#if 0
  if(thread0()) {
    print("mC\n");
    print(mC.shape()); print("\n"); print(mC.stride());
    print("\n\ngC\n");
    print(gC.shape()); print("\n"); print(gC.stride());
    print("\n\ntCsA\n");
    print(tCsA.shape()); print("\n"); print(tCsA.stride());
    print("\n\ntCsB\n");
    print(tCsB.shape()); print("\n"); print(tCsB.stride());
    print("\n\ntCgC\n");
    print(tCgC.shape()); print("\n"); print(tCgC.stride());
    print("\n\ntCrC\n");
    print(tCrC.shape()); print("\n"); print(tCrC.stride());
    print("\n\n");
  }
#endif

#if 1

    // TUTORIAL: Example of a very simple compute loop
    //   Data is read from global to shared memory via the tA|tB partitioning
    //   gemm(.) operates on the shared memory directly via the tC partitioning

    auto k_max = size<2>(tAgA);

    for (int k = 0; k < k_max; ++k) {
        // Copy gmem to smem
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);

        // In case copy uses cp.async, make sure that the cp.async
        // instructions are ordered with respect to other cp.async
        // instructions (fence), then wait on all the outstanding copy
        // operations (wait<0>()).  __syncthreads() alone does not do
        // this.
        //
        // NOTE: cp_async_wait<0>() currently issues cp.async.wait_all.
        // This is equivalent to cp.async.commit_group followed by
        // cp.async_wait_group 0.  This should make the first
        // cp_async_fence() (which also issues cp.async.commit_group)
        // redundant.  The tutorial works as-is, so we'll leave the
        // redundant fence in for now and study its removal later.
        cp_async_fence();
        cp_async_wait<0>();

        __syncthreads();

        // Compute gemm on smem
        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }

#endif

    //
    // Epilogue
    //

    axpby(alpha, tCrC, beta, tCgC);
}
)";

GemmGraph::GemmGraph(std::vector<Node *> operators_list,
                     std::vector<Data *> inputs_list,
                     std::vector<Data *> outputs_list, std::string name_value)
    : Graph(operators_list, inputs_list, outputs_list, name_value) {}

void GemmGraph::split(std::vector<size_t> thread_block_size,
                      std::vector<size_t> warp_size,
                      std::vector<size_t> thread_size) {
  // Check
  CHECK_EQ(thread_block_size.size(), 3);
  CHECK_EQ(warp_size.size(), 3);
  CHECK_EQ(thread_size.size(), 3);
  // Set
  this->thread_block_size = thread_block_size;
  this->warp_size = warp_size;
  this->thread_size = thread_size;
  // Split
  // std::vector<int64_t> split_dimension = {thread_block_size[0],
  //                                         thread_block_size[1]};
  // Split split(split_dimension);
  // tiles = outputs_list[0]->tiling(split);
}

void GemmGraph::applyPlatform(Platform platform) { this->platform = platform; }

std::string GemmGraph::generatorHead(int64_t indent = 0) {
  std::string result = "\n";
  result += platform.head();
  if (platform.type == Platform::CUDA) {
    result += "\n#include <cutlass/cutlass.h>\n";
    result += "#include <cutlass/gemm/device/gemm_splitk_parallel.h >\n";
    result += "#include <cutlass/util/host_tensor.h>\n";
    result += "#include <cutlass/util/reference/device/gemm.h>\n";
    result += "#include <cutlass/util/reference/host/tensor_compare.h>\n";
    result += "#include <cutlass/util/reference/host/tensor_copy.h>\n";
    result += "#include <cutlass/util/reference/host/tensor_fill.h>\n";
    result += "#include <cutlass/util/tensor_view_io.h>\n";
    result += "#include <helper.h>\n";
  }
  LOG(WARNING) << result;
  return result;
}

std::string GemmGraph::generatorTask(int64_t indent = 0) {
  std::string result = "\n";
  result += GemmKernel;
  return result;
}

std::string GemmGraph::generatorHost(int64_t indent = 0) {
  std::string result = "\n";
  // Add template parameters.
  result +=
      "template <typename TA, typename TB, typename TC, typename Alpha, "
      "typename Beta>";
  // Add function name.
  result += platform.globalFuncDecl(name + "_kernel");
  // Add function parameters.
  result +=
      "(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* "
      "B,\n",
      "int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {";
  if (platform.type == Platform::CUDA) {
    result += "using namespace cute;\n";
    result += "auto M = int(m);\n";
    result += "auto N = int(n);\n";
    result += "auto K = int(k);\n";

    // TODO: Generate based on the tiling.
    result += "auto dA = make_stride(Int<1>{}, ldA);";
    result += "auto dB = make_stride(Int<1>{}, ldB);";
    result += "auto dC = make_stride(Int<1>{}, ldC);";

    // TODO: Block sizes generate based on the tiling.
    // Define block sizes (static)
    result += "auto bM = Int<128>{};\n";
    result += "auto bN = Int<128>{};\n";
    result += "auto bK = Int<8>{};\n";

    // TODO:Block layouts generate based on the tiling.
    // Define block layouts (static)
    result += "auto sA = make_layout(make_shape(bM, bK));\n";
    result += "auto sB = make_layout(make_shape(bN, bK));\n";
    result += "auto sC = make_layout(make_shape(bM, bN));\n";

    // TODO: Thread layouts generate based on the tiling.
    // Define the thread layouts (static)
    result += "auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));\n";
    result += "auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));\n";
    result += "auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));\n";

    // Call GEMM kernel
    result += "dim3 dimBlock(size(tC));\n";
    result +=
        "dim3 dimGrid(ceil_div(size(M), size(bM)), ceil_div(size(N), "
        "size(bN)));\n";
    result += "gemm_device<<<dimGrid, dimBlock, 0, stream>>>(",
        "M, N, K, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha, "
        "beta);\n";
  }
  LOG(WARNING) << result;
  return result;
}

std::string GemmGraph::generatorCode(int64_t indent = 0) { return ""; }

std::string GemmGraph::generatorHeadFile(int64_t indent = 0) { return ""; }

std::string GemmGraph::generatorSourceFile(int64_t indent = 0) {
  std::string result;
  result += generatorHead();
  result += generatorTask();
  result += generatorHost();
  result += generatorCode();
  return result;
}
}  // namespace infini