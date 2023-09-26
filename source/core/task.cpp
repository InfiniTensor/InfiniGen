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

std::string SingleTask::generatorCode(Platform platform, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  if (core_list.empty()) {
    return "";
  }
  result += platform.deviceFuncDecl(name);
  result += "(" + getArguments() + ") {\n";

  result += indentation(indent + 1);
  result += "if (";
  for (int i = 0; i < core_list.size(); ++i) {
    result += platform.taskIdx() + " == " + std::to_string(core_list[i]);
    result += (i == core_list.size() - 1 ? ")" : " || ");
  }
  result += "{\n";

  result += indentation(indent + 2) + platform.regDecl("char", cache.name) +
            "[" + std::to_string(cache.cache_size) + "];\n";

  result += indentation(indent + 2) +
            platform.ldramDecl("char", cache.name + "_ldram") + "[" +
            std::to_string(cache.ldram_size) + "];\n";

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
                           TileTensor tiles, std::string name_value)
    : Task(cache_length, swap_length, align_length, cache_name, name_value),
      tile_tensor(tiles) {
  name =
      (name_value == "" ? "ParallelTask_" + std::to_string(index) : name_value);
}

std::string ParallelTask::generatorCode(Platform platform, int64_t indent = 0) {
  std::string result = "\n" + indentation(indent);
  result += platform.deviceFuncDecl(name);
  result += "(" + getArguments() + ") {\n";

  // TODO: delcare cache
  result += indentation(indent + 1) + platform.regDecl("char", cache.name) +
            "[" + std::to_string(cache.cache_size) + "];\n";

  result += indentation(indent + 1) +
            platform.ldramDecl("char", cache.name + "_ldram") + "[" +
            std::to_string(cache.ldram_size) + "];\n";

  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result, indent + 1);
  }

  result += indentation(indent) + "}\n";
  return result;
}

}  // namespace infini
