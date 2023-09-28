#include "graphs/binary_unary_graph.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"
#include "core/task.h"
#include "core/utils.h"
#include <algorithm>

namespace infini {

BinaryUnaryGraph::BinaryUnaryGraph(std::vector<Node *> operators_list,
                                   std::vector<Data *> inputs_list,
                                   std::vector<Data *> outputs_list,
                                   std::string name_value)
    : Graph(operators_list, inputs_list, outputs_list, name_value) {}

void BinaryUnaryGraph::applyPlatform(Platform platform) {
  this->platform = platform;
  for (auto data : inputs) {
    data->flatten();
  }
  for (auto data : outputs) {
    data->flatten();
  }
  for (auto data : temps) {
    data->flatten();
  }
  auto tiles = inputs[0]->tiling({1024});
  std::vector<Node *> sorted_op = topoSort();
  Task *task = nullptr;
  task = new ParallelTask(1024 * 10, 1024 * 100, 1024, "cache", tiles);
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
  for (int i = 0; i < sorted_op.size(); ++i) {
    Micro *micro = nullptr;
    if (platform == Platform::BANG) {
      micro = new BangAddMicro(
          sorted_op[i]->outputs[0]->name, sorted_op[i]->outputs[0]->data_offset,
          sorted_op[i]->inputs[0]->name, sorted_op[i]->inputs[0]->data_offset,
          sorted_op[i]->inputs[1]->name, sorted_op[i]->inputs[1]->data_offset,
          VECTOR_PRODUCT(tiles({0}).tile_dimension),
          sorted_op[i]->inputs[0]->tensor_datatype);
    } else if (platform == Platform::CUDA) {
      micro = new CudaAddMicro(
          sorted_op[i]->outputs[0]->name, sorted_op[i]->outputs[0]->data_offset,
          sorted_op[i]->inputs[0]->name, sorted_op[i]->inputs[0]->data_offset,
          sorted_op[i]->inputs[1]->name, sorted_op[i]->inputs[1]->data_offset,
          VECTOR_PRODUCT(tiles({0}).tile_dimension),
          sorted_op[i]->inputs[0]->tensor_datatype);
    }
    task->pushMicro(micro);

    // Update remain data
    for (auto input : sorted_op[i]->inputs) {
      temp_remain[input] -= 1;
      if (temp_remain[input] == 0) {
        temp_remain.erase(input);
        // Free
        if (platform == Platform::BANG) {
          micro = new BangFreeMicro(input->name, input->data_offset,
                                    VECTOR_PRODUCT(tiles({0}).tile_dimension),
                                    input->tensor_datatype);
        } else if (platform == Platform::CUDA) {
          micro = new CudaFreeMicro(input->name, input->data_offset,
                                    VECTOR_PRODUCT(tiles({0}).tile_dimension),
                                    input->tensor_datatype);
        }
        task->pushMicro(micro);
      }
    }

    // Store
    for (auto output : sorted_op[i]->outputs) {
      auto it = std::find(outputs.begin(), outputs.end(), output);
      if (it != outputs.end()) {
        if (platform == Platform::BANG) {
          micro = new BangStoreMicro(output->name, output->data_offset,
                                     VECTOR_PRODUCT(tiles({0}).tile_dimension),
                                     output->tensor_datatype);
        } else if (platform == Platform::CUDA) {
          micro = new CudaStoreMicro(output->name, output->data_offset,
                                     VECTOR_PRODUCT(tiles({0}).tile_dimension),
                                     output->tensor_datatype);
        }
        task->pushMicro(micro);
      }
    }
  }
  task_list.push_back(task);
}

std::string BinaryUnaryGraph::generatorHead(int64_t indent = 0) {
  // generate device function
  std::string result = "\n";
  result += platform.head();
  LOG(WARNING) << result;
  return result;
}

std::string BinaryUnaryGraph::generatorTask(int64_t indent = 0) {
  // generate device function
  std::string result = "\n";
  for (int i = 0; i < task_list.size(); i++) {
    result += task_list[i]->generatorCode(platform, indent);
  }
  LOG(WARNING) << result;
  return result;
}

std::string BinaryUnaryGraph::generatorHost(int64_t indent = 0) {
  // generate global function
  std::string result = "\n";
  result += platform.globalFuncDecl(name + "_kernel");

  std::vector<std::string> arguments_list;
  for (int i = 0; i < inputs.size(); ++i) {
    arguments_list.push_back(datatype_string(inputs[i]->tensor_datatype) +
                             " *" + inputs[i]->name);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    arguments_list.push_back(datatype_string(outputs[i]->tensor_datatype) +
                             " *" + outputs[i]->name);
  }
  std::string arguments = string_gather(arguments_list);

  result += "(" + arguments + ") {\n";
  result += indentation(indent + 1) + task_list[0]->name;
  result += "(" + task_list[0]->getArguments(false) + ");\n";
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

  // TODO
  std::string parallel_config = "dim, CNRT_FUNC_TYPE_UNION1, queue";
  std::string result = "\n" + indentation(indent);
  result += "void " + name + "(" + platform.queue() + " queue, " + arguments +
            ") {\n";
  result += indentation(indent + 1) + "cnrtDim3_t dim = {4,1,1};\n";
  result += indentation(indent + 1) + name + "_kernel";
  result += "<<<" + parallel_config + ">>>";
  result += "(" + operands + ");\n";
  result += indentation(indent) + "}\n";

  LOG(WARNING) << result;

  return result;
}

}  // namespace infini
