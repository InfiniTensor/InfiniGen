#pragma once
#include <vector>
#include "core/micro.h"
#include "core/cache.h"

namespace infini {

class Task {
 public:
  std::vector<Micro*> micro_list;
  Cache cache;

 public:
  // Constructor
  Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
       std::string cache_name);
  // Destructor
  ~Task() = default;
  // Function
  void pushMicro(Micro* micro);
  std::string generatorCode();
};

class ParallelTask {
 public:
  std::vector<Micro*> micro_list;
  Cache cache;
  int parallel;

 public:
  // Constructor
  ParallelTask(int64_t cache_length, int64_t swap_length, int64_t align_length,
               std::string cache_name, int64_t parallel_value);
  // Destructor
  ~ParallelTask() = default;
  // Function
  void pushMicro(Micro* micro);
  std::string generatorCode();
};

}  // namespace infini
