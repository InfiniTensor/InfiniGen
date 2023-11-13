#include "graphs/gemm_graph.h"

namespace infini {
GemmGraph::GemmGraph(std::vector<Node *> operators_list,
                     std::vector<Data *> inputs_list,
                     std::vector<Data *> outputs_list, std::string name_value)
    : Graph(operators_list, inputs_list, outputs_list, name_value) {}

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
}  // namespace infini