#include "core/utils.h"
#include "core/cache.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadCuda::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  int64_t length_in_bytes = 1 * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.load(cache_data);
  std::string length_string = std::to_string(length);
  std::string cache_string =
      "((" + datatype_string(data_type) + "*)" + "(" + cache.name + "))" + "[" +
      std::to_string(result.cache_offset / datatype_size(data_type));
  std::string data_string = name + "[" + std::to_string(offset) + " + ";
  data_string += platform.taskId() + " * " + length_string + " + ";

  if (result.location == CacheHitLocation::CACHE) {
    return cache_string;
  } else {
    if (result.location == CacheHitLocation::LDRAM) {
      // Since nvcc will do the management of register and local memory,
      // for cuda kernels, we set LDRAM size to 0, and we assume that hit
      // location will never be LDRAM. The same below.
      return "";
    } else if (result.location == CacheHitLocation::NOT_FOUND) {
      code += indentation(indent) + cache_string + "] = " + data_string +
              "threadIdx.x];\n";
      return cache_string;
    } else {
      return "";
    }
  }
}

std::string StoreCuda::generatorCode(Cache& cache, std::string& code,
                                     int64_t indent) {
  int64_t length_in_bytes = 1 * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.find(cache_data);
  std::string length_string = std::to_string(length);
  std::string cache_string =
      "((" + datatype_string(data_type) + "*)" + "(" + cache.name + "))" + "[" +
      std::to_string(result.cache_offset / datatype_size(data_type));
  std::string data_string = name + "[" + std::to_string(offset) + " + ";
  data_string += platform.taskId() + " * " + length_string + " + ";

  if (result.location == CacheHitLocation::CACHE) {
    code += indentation(indent) + data_string +
            "threadIdx.x] = " + cache_string + "];\n";
    return cache_string;
  } else if (result.location == CacheHitLocation::LDRAM) {
    return "";
  } else {
    return "";
  }
}

std::string AllocateCuda::generatorCode(Cache& cache, std::string& code,
                                        int64_t indent) {
  int64_t length_in_bytes = 1 * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.allocate(cache_data);
  std::string length_string = std::to_string(length);
  std::string cache_string =
      "((" + datatype_string(data_type) + "*)" + "(" + cache.name + "))" + "[" +
      std::to_string(result.cache_offset / datatype_size(data_type));
  std::string data_string = name + "[" + std::to_string(offset) + " + ";
  data_string += platform.taskId() + " * " + length_string + " + ";
  return cache_string;
}

std::string FreeCuda::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  int64_t length_in_bytes = 1 * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.free(cache_data);
  return "";
}

}  // namespace infini
