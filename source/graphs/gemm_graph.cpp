#include "graphs/gemm_graph.h"

namespace infini {
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

std::string GemmGraph::generatorHead(int64_t indent) {
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

std::string GemmGraph::generatorHost(int64_t indent = 0) {
  std::string result = "\n";
  result += platform.globalFuncDecl(name + "_kernel");
  if (platform.type == Platform::CUDA) {
    result += fmt::format("using ElementOutput = {}\n",
                          datatype_string(outputs[0]->tensor_datatype));
    result += fmt::format("using ElementAccumulator = {}\n",
                          datatype_string(outputs[0]->tensor_datatype));
    result += fmt::format("using ElementComputeEpilogue = {}\n",
                          datatype_string(outputs[0]->tensor_datatype));
    result += "using ElementInputA = cutlass::half_t;\n";
  }
  LOG(WARNING) << result;
  return result;
}
}  // namespace infini