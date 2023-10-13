#include "core/task.h"
#include "graphs/binary_unary_graph.h"
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
  tiles = inputs[0]->tiling({1024});
  std::vector<Node *> sorted_op = topoSort();
  Task *task = nullptr;
  task = new ParallelTask(1024 * 20, 1024 * 100, 4, "cache", tiles);
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
  MicroRegistry instance = MicroRegistry::getInstance();
  for (int i = 0; i < sorted_op.size(); ++i) {
    Micro *micro = nullptr;
    int64_t length = VECTOR_PRODUCT(tiles({0}).tile_dimension);
    TensorDatatype dtype = sorted_op[i]->inputs[0]->tensor_datatype;
    std::vector<OperandType> operands;
    for (auto output : sorted_op[i]->outputs) {
      operands.push_back(
          OperandType{output->name, output->data_offset, length, dtype});
    }
    for (auto input : sorted_op[i]->inputs) {
      operands.push_back(
          OperandType{input->name, input->data_offset, length, dtype});
    }
    micro = instance.getConstructor(MicroAttrs{
        sorted_op[i]->getOperatorType(), platform.underlying()})(operands);
    task->pushMicro(micro);
    // Update remain data
    for (auto input : sorted_op[i]->inputs) {
      temp_remain[input] -= 1;
      if (temp_remain[input] == 0) {
        temp_remain.erase(input);
        // Free
        micro = instance.getConstructor(
            MicroAttrs{OperatorType::FREE, platform.underlying()})(
            {OperandType{input->name, input->data_offset,
                         VECTOR_PRODUCT(tiles({0}).tile_dimension),
                         input->tensor_datatype}});
        task->pushMicro(micro);
      }
    }

    // Store
    for (auto output : sorted_op[i]->outputs) {
      auto it = std::find(outputs.begin(), outputs.end(), output);
      if (it != outputs.end()) {
        micro = instance.getConstructor(
            MicroAttrs{OperatorType::STORE, platform.underlying()})(
            {OperandType{output->name, output->data_offset,
                         VECTOR_PRODUCT(tiles({0}).tile_dimension),
                         output->tensor_datatype}});
        task->pushMicro(micro);
      }
    }
  }

  task_list.push_back(task);

  // Remainder part task
  // Assume 1-d
  if (!tiles.isNeat()) {
    Task *remainder_task = nullptr;
    remainder_task = new ParallelTask(1024 * 20, 1024 * 100, 4, "cache", tiles);
    for (auto data : inputs) {
      temp_remain[data] = data->remaining;
      remainder_task->addArgument(data->tensor_datatype, data->name);
    }
    for (auto data : temps) {
      temp_remain[data] = data->remaining;
    }
    for (auto data : outputs) {
      temp_remain[data] = data->remaining;
      remainder_task->addArgument(data->tensor_datatype, data->name);
    }
    int64_t offset =
        tiles.numNeatTiles() * VECTOR_PRODUCT(tiles.tiles[0].tile_dimension);
    for (int i = 0; i < sorted_op.size(); ++i) {
      Micro *remainder_micro = nullptr;

      int64_t length = VECTOR_PRODUCT(tiles.remain_tiles[0].tile_dimension);
      TensorDatatype dtype = sorted_op[i]->inputs[0]->tensor_datatype;
      std::vector<OperandType> operands;
      for (auto output : sorted_op[i]->outputs) {
        operands.push_back(OperandType{
            output->name, output->data_offset + offset, length, dtype});
      }
      for (auto input : sorted_op[i]->inputs) {
        operands.push_back(OperandType{input->name, input->data_offset + offset,
                                       length, dtype});
      }
      remainder_micro = instance.getConstructor(MicroAttrs{
          sorted_op[i]->getOperatorType(), platform.underlying()})(operands);
      remainder_task->pushMicro(remainder_micro);

      // Update remain data
      for (auto input : sorted_op[i]->inputs) {
        temp_remain[input] -= 1;
        if (temp_remain[input] == 0) {
          temp_remain.erase(input);
          // Free
          remainder_micro = instance.getConstructor(
              MicroAttrs{OperatorType::FREE, platform.underlying()})(
              {OperandType{input->name, input->data_offset + offset, length,
                           input->tensor_datatype}});
          remainder_task->pushMicro(remainder_micro);
        }
      }

      // Store
      for (auto output : sorted_op[i]->outputs) {
        auto it = std::find(outputs.begin(), outputs.end(), output);
        if (it != outputs.end()) {
          remainder_micro = instance.getConstructor(
              MicroAttrs{OperatorType::STORE, platform.underlying()})(
              {OperandType{output->name, output->data_offset + offset, length,
                           output->tensor_datatype}});
          remainder_task->pushMicro(remainder_micro);
        }
      }
    }
    task_list.push_back(remainder_task);
  }
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
  if (tiles.numNeatTiles() % 4 != 0) {
    result += indentation(indent + 1) + platform.workingCoreCond(tiles) + "\n";
  }
  result += indentation(indent + 1) + task_list[0]->name;
  result += "(" + task_list[0]->getArguments(false) + ");\n";

  if (!tiles.isNeat()) {
    result += indentation(indent + 1) + "if (" +
              platform.remainingTileCond(tiles) + ") {\n";
    result += indentation(indent + 2) + task_list[1]->name;
    result += "(" + task_list[1]->getArguments(false) + ");\n";
    result += indentation(indent + 1) + "}\n";
  }

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
  std::string result = "\n" + indentation(indent);
  result += "void " + name + "(" + platform.queue() + " queue, " + arguments +
            ") {\n";
  result += indentation(indent + 1) + platform.taskScaleDecl(tiles) + "\n";
  result += indentation(indent + 1) + name + "_kernel";
  result += platform.syntacticSugar();
  result += "(" + operands + ");\n";
  result += indentation(indent) + "}\n";

  LOG(WARNING) << result;

  return result;
}

std::string BinaryUnaryGraph::generatorHeadFile(int64_t indent = 0) {
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

  std::string result;

  result +=
      "void " + name + "(" + platform.queue() + " queue, " + arguments + ");\n";
  LOG(WARNING) << result;
  return result;
}

std::string BinaryUnaryGraph::generatorSourceFile(int64_t indent = 0) {
  std::string result;
  result += generatorHead();
  result += generatorTask();
  result += generatorHost();
  result += generatorCode();
  return result;
}

}  // namespace infini
