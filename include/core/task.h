#pragma once
#include <vector>
#include <utility>
#include "core/micro.h"
#include "core/cache.h"
#include "core/platform.h"

namespace infini {

class Task {
 public:
  std::string name;
  const int64_t index;
  std::vector<Micro *> micro_list;
  Cache cache;

  Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
       std::string cache_name, std::string name_value = "");
  ~Task() = default;

 protected:
  static int64_t count;
  std::vector<std::pair<std::string, std::string>> arguments;

 public:
  void pushMicro(Micro *micro);
  void addArgument(TensorDatatype type, std::string name);
  std::string getArguments(bool with_type);
  virtual std::string generatorCode(Platform platform, int64_t indent) = 0;
};

class SingleTask : public Task {
 public:
  std::vector<int64_t> core_list;

 public:
  // Constructor
  SingleTask(int64_t cache_length, int64_t swap_length, int64_t align_length,
             std::string cache_name, std::string name_value = "");
  // Destructor
  ~SingleTask() = default;
  // Function
  std::string generatorCode(Platform platform, int64_t indent) override;
  void dispatch(int64_t core);
};

class ParallelTask : public Task {
 public:
  int64_t parallel;

 public:
  // Constructor
  ParallelTask(int64_t cache_length, int64_t swap_length, int64_t align_length,
               std::string cache_name, int64_t parallel_value,
               std::string name_value = "");
  // Destructor
  ~ParallelTask() = default;
  // Function
  std::string generatorCode(Platform platform, int64_t indent) override;
};

}  // namespace infini
