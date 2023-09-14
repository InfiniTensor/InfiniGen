#include "graphs/binary_unary_graph.h"
#include "micros/binary_micro.h"
#include "core/task.h"
#include "core/utils.h"

namespace infini {

BinaryUnaryGraph::BinaryUnaryGraph(std::vector<Node*> operators_list,
                                   std::vector<Data*> inputs_list,
                                   std::vector<Data*> outputs_list,
                                   std::string name_value)
    : Graph(operators_list, inputs_list, outputs_list, name_value) {}

void BinaryUnaryGraph::generatorCode() {
  int64_t tensor_len = VECTOR_PRODUCT(inputs[0]->tensor_dimension);
  int64_t tile_len = 1024;
  int64_t loop = tensor_len / tile_len;
  int64_t rem_len = tensor_len % tile_len;
  std::vector<Node*> sorted_op = topoSort();
  for (int i = 0; i < loop; ++i) {
    LOG(INFO) << "======== Loop =========" << i;
    Task task(1024 * 10, 1024 * 100, 128, "cache");
    std::unordered_map<Data*, int64_t> temp_remain;
    for (auto data : inputs) {
      temp_remain[data] = data->remaining;
    }
    for (auto data : temps) {
      temp_remain[data] = data->remaining;
    }
    for (auto data : outputs) {
      temp_remain[data] = data->remaining;
    }
    for (auto op : sorted_op) {
      // TODO: codegen
      for (auto input : op->inputs) {
        temp_remain[input] -= 1;
        if (temp_remain[input] == 0) {
          temp_remain.erase(input);
        }
      }
      CudaAddMicro* micro = new CudaAddMicro(
          op->outputs[0]->name, i * tile_len, op->inputs[0]->name, i * tile_len,
          op->inputs[1]->name, i * tile_len, tile_len);
      task.pushMicro(micro);
      LOG(INFO) << "Codegen: " + op->name;
      std::string temp = "Remain: [";
      for (auto data : temp_remain) {
        temp += data.first->name;
        temp += ", ";
      }
      temp += "]";
      LOG(INFO) << temp;
    }
    LOG(WARNING) << task.generatorCode();
  }
}

}  // namespace infini
