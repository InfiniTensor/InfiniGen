#include <algorithm>
#include "core/worker.h"
#include "core/utils.h"

namespace infini {

Worker::Worker(int64_t num, int64_t total, int64_t align_size, std::string name,
               MemoryDispatch dispatch) {
  worker_name = name;
  cache_line_num = num;
  total_register_size = total;
  cache_line_size = PAD_DOWN(total / num, align_size);
  cache_remain_size = total_register_size - cache_line_size * cache_line_num;
  cache_line_next = 0;
  cache_dispatch = dispatch;
  for (auto i = 0; i < cache_line_num; ++i) {
    std::string cache_info = worker_name + "_cache" + std::to_string(i);
    std::string data_info = "null_offset_size";
    Cacheline temp(cache_info, data_info, 0);
    info.push_back(temp);
  }
}

void Worker::clearInfo() {
  cache_line_next = 0;
  for (auto i = 0; i < cache_line_num; ++i) {
    std::get<1>(info[i]) = "null_offset_size";
    std::get<2>(info[i]) = 0;
  }
}

void Worker::resetInfo(int64_t num, int64_t align_size) {
  cache_line_num = num;
  cache_line_size = PAD_DOWN(total_register_size / num, align_size);
  cache_remain_size = total_register_size - cache_line_size * cache_line_num;
  cache_line_next = 0;
  info.clear();
  for (auto i = 0; i < cache_line_num; ++i) {
    std::string cache_info = worker_name + "_cache" + std::to_string(i);
    std::string data_info = "null_offset_size";
    Cacheline temp(cache_info, data_info, 0);
    info.push_back(temp);
  }
}

void Worker::resetDispatch(MemoryDispatch dispatch) {
  this->clearInfo();
  cache_dispatch = dispatch;
}

std::string Worker::loadData(std::string data_info) {
  auto tokens = STRING_SPLIT(data_info, '_');
  std::string type = tokens[0];
  int64_t offset = std::stoll(tokens[1]);
  int64_t size = std::stoll(tokens[2]);
  if (size > total_register_size) {
    LOG(ERROR) << "Worker cache size is less than data size.";
    return "";
  }
  int64_t need_cache_num = DIV_UP(size, cache_line_size);
  if (need_cache_num > cache_line_num) {
    LOG(ERROR) << "Worker cache num is less than what is needed.";
    return "";
  }
  switch (cache_dispatch) {
    case MemoryDispatch::RANDOM:
    case MemoryDispatch::FIFO:
      return loadDataFIFO(type, offset, size);
    case MemoryDispatch::LRU:
      return loadDataLRU(type, offset, size);
    case MemoryDispatch::LFU:
      return loadDataLFU(type, offset, size);
    default:
      return loadDataFIFO(type, offset, size);
  }
}

void Worker::addSubordinate(Worker* worker) { subordinate.push_back(worker); }

void Worker::addSubordinates(std::vector<Worker*>& workers) {
  subordinate.insert(subordinate.end(), workers.begin(), workers.end());
}

void Worker::printInformation(int64_t level) {
  std::string info_string = std::string("    ") * level;
  info_string += "—— Worker ";
  info_string += "Name: ";
  info_string += worker_name;
  info_string += " ";
  info_string += "Total Size: ";
  info_string += std::to_string(total_register_size);
  info_string += " ";
  info_string += "Cache Size: ";
  info_string += std::to_string(cache_line_size);
  info_string += " ";
  info_string += "Cache Num: ";
  info_string += std::to_string(cache_line_num);
  info_string += " ";
  info_string += "Remain Size: ";
  info_string += std::to_string(cache_remain_size);
  info_string += " ";
  info_string += "Next Line: ";
  info_string += std::to_string(cache_line_next);
  info_string += " ";
  info_string += "Memory Dispatch: ";
  info_string += TO_STRING(cache_dispatch);
  LOG(INFO) << info_string;
  for (int i = 0; i < subordinate.size(); ++i) {
    subordinate[i]->printInformation(level + 1);
  }
}

void Worker::printSummary(int64_t level) {
  std::string info_string = std::string("    ") * level;
  info_string += "Worker ";
  info_string += "Total: ";
  info_string += std::to_string(total_register_size);
  info_string += " ";
  info_string += "Cache Size: ";
  info_string += std::to_string(cache_line_size);
  info_string += " ";
  info_string += "Cache Num: ";
  info_string += std::to_string(cache_line_num);
  info_string += " ";
  info_string += "Memory Dispatch: ";
  info_string += TO_STRING(cache_dispatch);
  info_string += "\n";
  LOG(PURE) << info_string;
  for (int i = 0; i < subordinate.size(); ++i) {
    subordinate[i]->printSummary(level + 1);
  }
}

std::string Worker::loadDataFIFO(std::string type, int64_t offset,
                                 int64_t size) {
  int64_t need_cache_num = DIV_UP(size, cache_line_size);
  bool match = false;
  std::string result = "";
  for (auto i = 0; i < need_cache_num; ++i) {
    match = false;
    size -= cache_line_size;
    int64_t data_size = (size >= 0 ? cache_line_size : size + cache_line_size);
    // FIND
    std::string data_info = type + "_" +
                            std::to_string(offset + i * cache_line_size) + "_" +
                            std::to_string(data_size);
    for (auto i = 0; i < cache_line_num; ++i) {
      if (std::get<1>(info[i]) == data_info) {
        // Cache match
        match = true;
        LOG(INFO) << "Worker cache match, " + data_info +
                         " has been cached on " + std::get<0>(info[i]);
        result += data_info + " cached " + std::get<0>(info[i]) + " ";
        break;
      }
    }
    if (!match) {
      // Cache mismatch
      std::get<1>(info[cache_line_next]) = data_info;
      LOG(INFO) << "Worker cache mismatch, " + data_info +
                       " will be cached on " +
                       std::get<0>(info[cache_line_next]);
      result +=
          data_info + " moved " + std::get<0>(info[cache_line_next]) + " ";
      // UPDATE
      ++cache_line_next;
      cache_line_next %= cache_line_num;
    }
  }
  return result;
}

std::string Worker::loadDataLRU(std::string type, int64_t offset,
                                int64_t size) {
  int64_t need_cache_num = DIV_UP(size, cache_line_size);
  bool match = false;
  std::string result = "";
  for (auto i = 0; i < need_cache_num; ++i) {
    match = false;
    size -= cache_line_size;
    int64_t data_size = (size >= 0 ? cache_line_size : size + cache_line_size);
    // FIND
    std::string data_info = type + "_" +
                            std::to_string(offset + i * cache_line_size) + "_" +
                            std::to_string(data_size);
    for (auto i = 0; i < cache_line_num; ++i) {
      if (std::get<1>(info[i]) == data_info) {
        // Cache match
        match = true;
        std::get<2>(info[i]) = 0;
        LOG(INFO) << "Worker cache match, " + data_info +
                         " has been cached on " + std::get<0>(info[i]);
        result += data_info + " cached " + std::get<0>(info[i]) + " ";
      } else {
        ++std::get<2>(info[i]);
      }
    }
    if (!match) {
      // Catch mismatch
      std::get<1>(info[cache_line_next]) = data_info;
      std::get<2>(info[cache_line_next]) = 0;
      LOG(INFO) << "Worker cache mismatch, " + data_info +
                       " will be cached on " +
                       std::get<0>(info[cache_line_next]);
      result +=
          data_info + " moved " + std::get<0>(info[cache_line_next]) + " ";
    }
    // UPDATE
    cache_line_next = std::max_element(info.begin(), info.end()) - info.begin();
  }
  return result;
}

std::string Worker::loadDataLFU(std::string type, int64_t offset,
                                int64_t size) {
  int64_t need_cache_num = DIV_UP(size, cache_line_size);
  bool match = false;
  std::string result = "";
  for (auto i = 0; i < need_cache_num; ++i) {
    match = false;
    size -= cache_line_size;
    int64_t data_size = (size >= 0 ? cache_line_size : size + cache_line_size);
    // FIND
    std::string data_info = type + "_" +
                            std::to_string(offset + i * cache_line_size) + "_" +
                            std::to_string(data_size);
    for (auto i = 0; i < cache_line_num; ++i) {
      if (std::get<1>(info[i]) == data_info) {
        // Cache match
        match = true;
        ++std::get<2>(info[i]);
        LOG(INFO) << "Worker cache match, " + data_info +
                         " has been cached on " + std::get<0>(info[i]);
        result += data_info + " cached " + std::get<0>(info[i]) + " ";
      }
    }
    if (!match) {
      // Catch mismatch
      std::get<1>(info[cache_line_next]) = data_info;
      std::get<2>(info[cache_line_next]) = 1;
      LOG(INFO) << "Worker cache mismatch, " + data_info +
                       " will be cached on " +
                       std::get<0>(info[cache_line_next]);
      result +=
          data_info + " moved " + std::get<0>(info[cache_line_next]) + " ";
    }
    // UPDATE
    cache_line_next = std::min_element(info.begin(), info.end()) - info.begin();
  }
  return result;
}

std::string Worker::generatorBoneOnBANG(std::string cache_name, int64_t num) {
  std::string result = "";
  result += indentation(num) + cache_name + " buffer[" +
            std::to_string(total_register_size) + "];\n";
  for (auto i = 0; i < cache_line_num; ++i) {
    result += indentation(num) + "char *buffer_slice_" + std::to_string(i) +
              " = buffer + " + std::to_string(i * cache_line_size) + ";\n";
  }
  return result;
}

std::string Worker::generatorBoneOnCUDA(std::string cache_name, int64_t num) {
  return "cuda";
}

}  // namespace infini
