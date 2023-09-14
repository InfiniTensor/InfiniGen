#include "core/task.h"
#include "core/utils.h"

namespace infini {

Task::Task(int64_t cache_length, int64_t swap_length, int64_t align_length,
           std::string cache_name)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU) {}

void Task::pushMicro(Micro* micro) { micro_list.push_back(micro); }

std::string Task::generatorCode() {
  std::string result = "\n";
  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result);
  }
  return result;
}

//////////////////////////////////////////////////////////////////////

ParallelTask::ParallelTask(int64_t cache_length, int64_t swap_length,
                           int64_t align_length, std::string cache_name,
                           int64_t parallel_value)
    : cache(cache_length, swap_length, align_length, cache_name,
            MemoryDispatch::LRU),
      parallel(parallel_value) {}

void ParallelTask::pushMicro(Micro* micro) { micro_list.push_back(micro); }

std::string ParallelTask::generatorCode() {
  std::string result = "\n";
  for (int i = 0; i < micro_list.size(); ++i) {
    micro_list[i]->generatorCode(cache, result, "blockIdx.x");
  }
  return result;
}

}  // namespace infini
