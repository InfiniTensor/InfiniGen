#include "core/task.h"
#include "core/utils.h"

namespace infini {

int64_t Task::count = 0;

Task::Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
           std::string cache_name, std::string name_value)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU),
      name(name_value),
      index(count++) {}

void Task::pushMicro(Micro *micro) { micro_list.push_back(micro); }

void Task::addArgument(TensorDatatype type, std::string name) {
  arguments.push_back(std::make_pair(datatype_string(type), name));
}

std::string Task::getArguments(bool with_type = true) {
  std::vector<std::string> args;
  for (int i = 0; i < arguments.size(); i++) {
    if (with_type) {
      args.push_back(arguments[i].first + " *" + arguments[i].second);
    } else {
      args.push_back(arguments[i].second);
    }
  }
  return string_gather(args);
}

//////////////////////////////////////////////////////////////////////

SingleTask::SingleTask(int64_t cache_length, int64_t swap_length,
                       int64_t align_length, std::string cache_name,
                       std::string name_value)
    : Task(cache_length, swap_length, align_length, cache_name, name_value) {
  name =
      (name_value == "" ? "SingleTask_" + std::to_string(index) : name_value);
}

std::string SingleTask::generatorCode(PlatformType type, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  if (core_list.empty()) {
    return "";
  }
  if (type == PlatformType::BANG) {
    result += "__mlu_func__ void ";
  } else if (type == PlatformType::CUDA) {
    result += "__device__ void ";
  }
  result += name + "(" + getArguments() + ") {\n" + indentation(indent + 1);

  if (type == PlatformType::BANG) {
    result += "if (";
    for (int i = 0; i < core_list.size(); ++i) {
      result += "taskId == " + std::to_string(core_list[i]);
      result += (i == core_list.size() - 1 ? ")" : " || ");
    }
    result += "{\n";
  } else if (type == PlatformType::CUDA) {
    result += "if (";
    for (int i = 0; i < core_list.size(); ++i) {
      result += "blockId == " + std::to_string(core_list[i]);
      result += (i == core_list.size() - 1 ? ")" : " || ");
    }
    result += "{\n";
  }
  if (type == PlatformType::BANG) {
    result += indentation(indent + 2) + "__nram__ ";
  }
  result += indentation(indent + 2) + "char " + cache.name + "[" +
            std::to_string(cache.cache_size) + "];\n";
  if (type == PlatformType::BANG) {
    result += indentation(indent + 2) + "__ldram__ char " + cache.name +
              "_ldram[" + std::to_string(cache.ldram_size) + "];\n";
  }

  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result, indent + 2);
  }
  result += indentation(indent + 1) + "}\n";
  result += indentation(indent) + "}\n";
  return result;
}

void SingleTask::dispatch(int64_t core) { core_list.push_back(core); }

//////////////////////////////////////////////////////////////////////

ParallelTask::ParallelTask(int64_t cache_length, int64_t swap_length,
                           int64_t align_length, std::string cache_name,
                           int64_t parallel_value, std::string name_value)
    : Task(cache_length, swap_length, align_length, cache_name, name_value),
      parallel(parallel_value) {
  name =
      (name_value == "" ? "ParallelTask_" + std::to_string(index) : name_value);
}

std::string ParallelTask::generatorCode(PlatformType type, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  if (type == PlatformType::BANG) {
    result += "__mlu_func__ void ";
  } else if (type == PlatformType::CUDA) {
    result += "__device__ void ";
  }
  result += name + "(" + getArguments() + ") {\n" + indentation(indent + 1);

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
    micro_list[i]->generatorCode(cache, result, indent + 1);
  }

  result += indentation(indent) + "}\n";
  return result;
}

}  // namespace infini
