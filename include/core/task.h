#pragma once
#include <vector>
#include "core/micro.h"
#include "core/cache.h"
#include "core/graph.h"

namespace infini {

class Task {
 public:
  std::vector<Micro *> micro_list;
  Cache cache;

 public:
  // Constructor
  Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
       std::string cache_name);
  // Destructor
  ~Task() = default;
  // Function
  void pushMicro(Micro *micro);
  std::string generatorCode(int64_t indent);
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
  std::string arguments;
  std::string data_type;

 public:
  // Constructor
  ParallelTask(int64_t cache_length, int64_t swap_length, int64_t align_length,
               std::string cache_name, int64_t parallel_value);
  // Destructor
  ~ParallelTask() = default;
  // Function
  void pushMicro(Micro *micro);
  void setArguments(std::string arguments);
  void setDataType(TensorDatatype data_type);
  std::string generatorCode(PlatformType type, int64_t indent);
};

}  // namespace infini
