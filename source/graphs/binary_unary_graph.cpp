#include "graphs/binary_unary_graph.h"
#include "micros/binary_micro.h"
#include "core/task.h"
#include "core/utils.h"

namespace infini {

BinaryUnaryGraph::BinaryUnaryGraph(std::vector<Node *> operators_list,
                                   std::vector<Data *> inputs_list,
                                   std::vector<Data *> outputs_list,
                                   std::string name_value)
    : Graph(operators_list, inputs_list, outputs_list, name_value) {}

std::string BinaryUnaryGraph::generatorCode(PlatformType type,
                                            int64_t indent = 0) {
  int64_t tensor_len = VECTOR_PRODUCT(inputs[0]->tensor_dimension);
  int64_t tile_len = 1;
  int64_t loop = tensor_len / tile_len;
  int64_t rem_len = tensor_len % tile_len;
  std::vector<Node *> sorted_op = topoSort();
  LOG(INFO) << "======== Parallel =========" << loop;
  ParallelTask task(10, 100, 1, "cache", loop);
  std::unordered_map<Data *, int64_t> temp_remain;
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
    Micro *micro = nullptr;
    if (type == PlatformType::BANG) {
      micro = new BangAddMicro(op->outputs[0]->name, 0, op->inputs[0]->name, 0,
                               op->inputs[1]->name, 0, tile_len);
    } else if (type == PlatformType::CUDA) {
      micro = new CudaAddMicro(op->outputs[0]->name, 0, op->inputs[0]->name, 0,
                               op->inputs[1]->name, 0, tile_len);
    }
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

  std::string arguments = "";
  std::string operands = "";
  for (int i = 0; i < inputs.size(); ++i) {
    task.addArgument(inputs[i]->tensor_datatype, inputs[i]->name);

    arguments += datatype_string(inputs[i]->tensor_datatype);
    arguments += " *";
    arguments += inputs[i]->name;
    arguments += ", ";

    operands += inputs[i]->name;
    operands += ", ";
  }
  for (int i = 0; i < outputs.size(); ++i) {
    task.addArgument(outputs[i]->tensor_datatype, outputs[i]->name);

    arguments += datatype_string(outputs[i]->tensor_datatype);
    arguments += " *";
    arguments += outputs[i]->name;
    arguments += (i == (outputs.size() - 1) ? "" : ", ");

    operands += outputs[i]->name;
    operands += (i == (outputs.size() - 1) ? "" : ", ");
  }
  std::string result = task.generatorCode(type, indent);

  // generate global function
  result += "\n" + indentation(indent);
  if (type == PlatformType::BANG) {
    result += "__mlu_entry__ void ";
  } else if (type == PlatformType::CUDA) {
    result += "__global__ void ";
  }

  result += task.name + "_kernel(" + arguments + ") {\n";
  result += indentation(indent + 1) + task.name;
  result += "(" + operands + ");\n";
  result += indentation(indent) + "}\n";

  // TODO: generate final wrapper
  // TODO: tune parameters
  std::string parallel_config = "1, 64";
  result += "\n" + indentation(indent);
  result += "void " + name + "(" + arguments + ") {\n";
  result += indentation(indent + 1) + task.name + "_kernel";
  result += "<<<" + parallel_config + ">>>";
  result += "(" + operands + ");\n";
  result += indentation(indent) + "}\n";

  LOG(WARNING) << result;

  return result;
}

}  // namespace infini
