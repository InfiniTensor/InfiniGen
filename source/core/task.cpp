#include "core/task.h"
#include "core/utils.h"
#include "core/graph.h"

namespace infini {

int64_t ParallelTask::count = 0;

Task::Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
           std::string cache_name)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU) {}

void Task::pushMicro(Micro* micro) { micro_list.push_back(micro); }

std::string Task::generatorCode(int64_t indent = 0) {
  std::string result = "\n";
  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result, "", indent);
  }
  return result;
}

//////////////////////////////////////////////////////////////////////

ParallelTask::ParallelTask(int64_t cache_length, int64_t swap_length,
                           int64_t align_length, std::string cache_name,
                           int64_t parallel_value)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU),
      parallel(parallel_value),
      index(count++) {
  name = (name == "" ? "Task_" + std::to_string(index) : name);
}

void ParallelTask::pushMicro(Micro* micro) { micro_list.push_back(micro); }

void ParallelTask::setInputs(std::vector<Data*> tensors) { inputs = tensors; }

void ParallelTask::setOutputs(std::vector<Data*> tensors) { outputs = tensors; }

std::string ParallelTask::generatorCode(PlatformType type, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  if (type == PlatformType::BANG) {
    result += "__mlu_entry__ void ";
  } else if (type == PlatformType::CUDA) {
    result += "__device__ void ";
  }
  result += name + "_kernel";

  std::string arguments = "";
  std::string parameters = "";
  for (int i = 0; i < inputs.size(); ++i) {
    arguments += datatype_string(inputs[i]->tensor_datatype);
    arguments += " *";
    arguments += inputs[i]->name;
    arguments += ", ";

    parameters += inputs[i]->name;
    parameters += ", ";
  }
  for (int i = 0; i < outputs.size(); ++i) {
    arguments += datatype_string(outputs[i]->tensor_datatype);
    arguments += " *";
    arguments += outputs[i]->name;
    arguments += (i == (outputs.size() - 1) ? "" : ", ");

    parameters += outputs[i]->name;
    parameters += (i == (outputs.size() - 1) ? "" : ", ");
  }
  result += "(" + arguments + ") {";
  result += "\n" + indentation(indent + 1);

  // TODO: delcare cache
  std::string data_type = datatype_string(inputs[0]->tensor_datatype);
  result += data_type + " " + cache.name + "[" +
            std::to_string(cache.cache_size) + "];\n";
  if (type == PlatformType::BANG) {
    result += indentation(indent + 1) + data_type + " *" + cache.name +
              "_ldram[" + std::to_string(cache.cache_size) + "];\n";
  }

  for (int i = 0; i < micro_list.size(); ++i) {
    result += micro_list[i]->generatorCode(cache, result, "taskId", indent + 1);
  }

  result += indentation(indent) + "}\n";
  return result;
}

}  // namespace infini
