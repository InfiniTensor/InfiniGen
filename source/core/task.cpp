#include "core/task.h"
#include "core/utils.h"

namespace infini {

int64_t Task::count = 0;
int64_t ParallelTask::count = 0;

Task::Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
           std::string cache_name, std::string name_value)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU),
      index(count++) {
  name = (name_value == "" ? "Task_" + std::to_string(index) : name_value);
}

void Task::pushMicro(Micro *micro) { micro_list.push_back(micro); }

void Task::addArgument(TensorDatatype type, std::string name) {
  arguments.push_back(std::make_pair(datatype_string(type), name));
}

std::string Task::generatorCode(PlatformType type, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  if (type == PlatformType::BANG) {
    result += "__mlu_func__ void ";
  } else if (type == PlatformType::CUDA) {
    result += "__device__ void ";
  }
  result += name + "(";
  for (int i = 0; i < arguments.size(); i++) {
    result += arguments[i].first + " *" + arguments[i].second;
    result += (i == (arguments.size() - 1) ? "" : ", ");
  }
  result += ") {\n" + indentation(indent + 1);

  // TODO: delcare cache
  if (type == PlatformType::BANG) {
    result += "__nram__ ";
  }
  result +=
      "char " + cache.name + "[" + std::to_string(cache.cache_size) + "];\n";
  if (type == PlatformType::BANG) {
    result += indentation(indent + 1) + "__ldram__ char " + cache.name +
              "_ldram[" + std::to_string(cache.ldram_size) + "];\n";
  }

  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result, "taskId", indent + 1);
  }

  result += indentation(indent) + "}\n";
  return result;
}

//////////////////////////////////////////////////////////////////////

ParallelTask::ParallelTask(int64_t cache_length, int64_t swap_length,
                           int64_t align_length, std::string cache_name,
                           int64_t parallel_value, std::string name_value)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU),
      parallel(parallel_value),
      index(count++) {
  name =
      (name_value == "" ? "ParallelTask_" + std::to_string(index) : name_value);
}

void ParallelTask::pushMicro(Micro *micro) { micro_list.push_back(micro); }

void ParallelTask::addArgument(TensorDatatype type, std::string name) {
  arguments.push_back(std::make_pair(datatype_string(type), name));
}

std::string ParallelTask::generatorCode(PlatformType type, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  if (type == PlatformType::BANG) {
    result += "__mlu_func__ void ";
  } else if (type == PlatformType::CUDA) {
    result += "__device__ void ";
  }
  result += name + "(";
  for (int i = 0; i < arguments.size(); i++) {
    result += arguments[i].first + " *" + arguments[i].second;
    result += (i == (arguments.size() - 1) ? "" : ", ");
  }
  result += ") {\n" + indentation(indent + 1);

  // TODO: delcare cache
  if (type == PlatformType::BANG) {
    result += "__nram__ ";
  }
  result +=
      "char " + cache.name + "[" + std::to_string(cache.cache_size) + "];\n";
  if (type == PlatformType::BANG) {
    result += indentation(indent + 1) + "__ldram__ char " + cache.name +
              "_ldram[" + std::to_string(cache.ldram_size) + "];\n";
  }

  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result, "taskId", indent + 1);
  }

  result += indentation(indent) + "}\n";
  return result;
}

}  // namespace infini
