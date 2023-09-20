#pragma once
#include <vector>
#include <utility>
#include "core/micro.h"
#include "core/cache.h"

namespace infini {

class Task {
 public:
  std::string name;
  const int64_t index;
  std::vector<Micro *> micro_list;
  Cache cache;

 private:
  static int64_t count;
  // argument list, in the form of (type, name)
  std::vector<std::pair<std::string, std::string>> arguments;

 public:
  // Constructor
  Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
       std::string cache_name, std::string name_value = "");
  // Destructor
  ~Task() = default;
  // Function
  void pushMicro(Micro *micro);
  void addArgument(TensorDatatype type, std::string name);
  std::string generatorCode(PlatformType type, int64_t indent);
};

class ParallelTask {
 public:
  std::string name;
  const int64_t index;
  std::vector<Micro *> micro_list;
  Cache cache;
  int parallel;

 private:
  static int64_t count;
  // argument list, in the form of (type, name)
  std::vector<std::pair<std::string, std::string>> arguments;

 public:
  // Constructor
  ParallelTask(int64_t cache_length, int64_t swap_length, int64_t align_length,
               std::string cache_name, int64_t parallel_value,
               std::string name_value = "");
  // Destructor
  ~ParallelTask() = default;
  // Function
  void pushMicro(Micro *micro);
  void addArgument(TensorDatatype type, std::string name);
  std::string generatorCode(PlatformType type, int64_t indent);
};

}  // namespace infini
