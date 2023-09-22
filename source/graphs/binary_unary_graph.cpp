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

void BinaryUnaryGraph::applyPlatform(PlatformType type) {
  platform = type;
  int64_t tensor_len = VECTOR_PRODUCT(inputs[0]->tensor_dimension);
  int64_t tile_len = 1024;
  int64_t loop = tensor_len / tile_len;
  int64_t rem_len = tensor_len % tile_len;
  std::vector<Node *> sorted_op = topoSort();
  Task *task = nullptr;
  task = new ParallelTask(1024 * 10, 1024 * 100, 1024, "cache");
  std::unordered_map<Data *, int64_t> temp_remain;
  for (auto data : inputs) {
    temp_remain[data] = data->remaining;
    task->addArgument(data->tensor_datatype, data->name);
  }
  for (auto data : temps) {
    temp_remain[data] = data->remaining;
  }
  for (auto data : outputs) {
    temp_remain[data] = data->remaining;
    task->addArgument(data->tensor_datatype, data->name);
  }
  for (auto op : sorted_op) {
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
    task->pushMicro(micro);
  }
  task_list.push_back(task);
}

std::string BinaryUnaryGraph::generatorTask(int64_t indent = 0) {
  // generate device function
  std::string result = "";
  for (int i = 0; i < task_list.size(); i++) {
    result += task_list[i]->generatorCode(platform, indent);
  }
  LOG(WARNING) << result;
  return result;
}

std::string BinaryUnaryGraph::generatorHost(int64_t indent = 0) {
  // generate global function
  std::string result = "\n";
  if (platform == PlatformType::BANG) {
    result += "__mlu_entry__ void ";
  } else if (platform == PlatformType::CUDA) {
    result += "__global__ void ";
  }

  std::vector<std::string> arguments_list;
  std::vector<std::string> operands_list;
  for (int i = 0; i < inputs.size(); ++i) {
    arguments_list.push_back(datatype_string(inputs[i]->tensor_datatype) +
                             " *" + inputs[i]->name);
    operands_list.push_back(inputs[i]->name);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    arguments_list.push_back(datatype_string(outputs[i]->tensor_datatype) +
                             " *" + outputs[i]->name);
    operands_list.push_back(outputs[i]->name);
  }
  std::string arguments = string_gather(arguments_list);
  std::string operands = string_gather(operands_list);

  result += name + "_kernel(" + arguments + ") {\n";
  result += indentation(indent + 1) + task_list[0]->name;
  result += "(" + operands + ");\n";
  result += indentation(indent) + "}\n";

  LOG(WARNING) << result;
  return result;
}

std::string BinaryUnaryGraph::generatorCode(int64_t indent = 0) {
  std::vector<std::string> arguments_list;
  std::vector<std::string> operands_list;
  for (int i = 0; i < inputs.size(); ++i) {
    task_list[0]->addArgument(inputs[i]->tensor_datatype, inputs[i]->name);
    arguments_list.push_back(datatype_string(inputs[i]->tensor_datatype) +
                             " *" + inputs[i]->name);
    operands_list.push_back(inputs[i]->name);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    task_list[0]->addArgument(outputs[i]->tensor_datatype, outputs[i]->name);
    arguments_list.push_back(datatype_string(outputs[i]->tensor_datatype) +
                             " *" + outputs[i]->name);
    operands_list.push_back(outputs[i]->name);
  }
  std::string arguments = string_gather(arguments_list);
  std::string operands = string_gather(operands_list);

  std::string parallel_config = "1, 64";
  std::string result = "\n" + indentation(indent);
  result += "void " + name + "(" + arguments + ") {\n";
  result += indentation(indent + 1) + task_list[0]->name + "_kernel";
  result += "<<<" + parallel_config + ">>>";
  result += "(" + operands + ");\n";
  result += indentation(indent) + "}\n";

  LOG(WARNING) << result;

  return result;
}

}  // namespace infini
