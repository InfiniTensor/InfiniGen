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

  Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
       std::string cache_name, std::string name_value = "");
  ~Task() = default;

 protected:
  static int64_t count;
  std::vector<std::pair<std::string, std::string>> arguments;

 public:
  virtual void pushMicro(Micro *micro) = 0;
  virtual void addArgument(TensorDatatype type, std::string name) = 0;
  virtual std::string generatorCode(PlatformType type, int64_t indent) = 0;
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
  void pushMicro(Micro *micro) override;
  void addArgument(TensorDatatype type, std::string name) override;
  std::string generatorCode(PlatformType type, int64_t indent) override;
  void dispatch(int64_t core);
};

class ParallelTask : public Task {
 public:
  // Constructor
  ParallelTask(int64_t cache_length, int64_t swap_length, int64_t align_length,
               std::string cache_name, std::string name_value = "");
  // Destructor
  ~ParallelTask() = default;
  // Function
  void pushMicro(Micro *micro) override;
  void addArgument(TensorDatatype type, std::string name) override;
  std::string generatorCode(PlatformType type, int64_t indent) override;
};

}  // namespace infini
