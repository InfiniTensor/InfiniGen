#pragma once
#include "core/type.h"
#include <vector>

namespace infini {

class Worker {
 public:
  // Self information
  int64_t total_register_size;
  int64_t cache_line_size;
  int64_t cache_line_num;
  int64_t cache_remain_size;
  int64_t cache_line_next;
  MemoryDispatch cache_dispatch;
  std::string worker_name;
  std::vector<Cacheline> info;
  std::vector<Worker*> subordinate;

 public:
  // Constructor
  Worker() = delete;
  Worker(int64_t num, int64_t total, int64_t align_size, std::string name,
         MemoryDispatch dispatch);
  // Destructor
  ~Worker() = default;
  // Clear cache information
  void clearInfo();
  // Reset cache information
  void resetInfo(int64_t num, int64_t align_size);
  // Reset cache dispatch algorithm
  void resetDispatch(MemoryDispatch dispatch);
  // Load data
  void loadData(std::string datainfo);
  // Add subordinate
  void addSubordinate(Worker* worker);
  void addSubordinates(std::vector<Worker*>& workers);
  // Information
  void printInformation(int64_t level = 0);
  void printSummary(int64_t level = 0);

 private:
  // FIFO algorithm
  void loadDataFIFO(std::string type, int64_t offset, int64_t size);
  // LRU algorithm
  void loadDataLRU(std::string type, int64_t offset, int64_t size);
  // LFU algorithm
  void loadDataLFU(std::string type, int64_t offset, int64_t size);
};

}  // namespace infini
