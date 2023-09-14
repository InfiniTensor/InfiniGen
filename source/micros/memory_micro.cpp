
#include "micros/memory_micro.h"
#include "core/cache.h"

namespace infini {

std::string BangLoadMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.load(cache_data);
  std::string cache_string =
      cache.name + " + " + std::to_string(result.cache_offset);
  std::string data_string = data_name + " + " + std::to_string(data);
  std::string length_string = std::to_string(length);
  std::string ldram_from_string =
      cache.name + " + " + std::to_string(result.ldram_from_offset);
  int64_t replaced_data_count = result.ldram_to_offset.size();
  switch (result.location) {
    case CacheHitLocation::CACHE:
      return cache_string;
    case CacheHitLocation::LDRAM:
      if (replaced_data_count > 0) {
        for (int i = 0; i < replaced_data_count; i++) {
          std::string cache_from_string =
              cache.name + " + " +
              std::to_string(result.replaced_data_cache_offset[i]);
          std::string ldram_to_string =
              cache.name + " + " + std::to_string(result.ldram_to_offset[i]);
          std::string replaced_data_length_string =
              std::to_string(result.replaced_data_size[i]);
          code += "__memcpy(" + ldram_to_string + ", " + cache_from_string +
                  ", " + replaced_data_length_string + ", NRAM2LDRAM);\n";
        }
      }
      code += "__memcpy(" + cache_string + ", " + ldram_from_string + ", " +
              length_string + ", LDRAM2NRAM);\n";
      return cache_string;
    case CacheHitLocation::NOT_FOUND:
      if (replaced_data_count > 0) {
        for (int i = 0; i < replaced_data_count; i++) {
          std::string cache_from_string =
              cache.name + " + " +
              std::to_string(result.replaced_data_cache_offset[i]);
          std::string ldram_to_string =
              cache.name + " + " + std::to_string(result.ldram_to_offset[i]);
          std::string replaced_data_length_string =
              std::to_string(result.replaced_data_size[i]);
          code += "__memcpy(" + ldram_to_string + ", " + cache_from_string +
                  ", " + replaced_data_length_string + ", NRAM2LDRAM);\n";
        }
      }
      code += "__memcpy(" + cache_string + ", " + data_string + ", " +
              length_string + ", GDRAM2NRAM);\n";
      return cache_string;
    default:
      return "";
  }
}

std::string BangStoreMicro::generatorCode(Cache& cache, std::string& code) {
  return "";
}

std::string BangAllocateMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.load(cache_data);
  std::string cache_string =
      cache.name + " + " + std::to_string(result.cache_offset);
  std::string data_string = data_name + " + " + std::to_string(data);
  std::string length_string = std::to_string(length);
  std::string ldram_from_string =
      cache.name + " + " + std::to_string(result.ldram_from_offset);
  int64_t replaced_data_count = result.ldram_to_offset.size();
  if (replaced_data_count > 0) {
    for (int i = 0; i < replaced_data_count; i++) {
      std::string cache_from_string =
          cache.name + " + " +
          std::to_string(result.replaced_data_cache_offset[i]);
      std::string ldram_to_string =
          cache.name + " + " + std::to_string(result.ldram_to_offset[i]);
      std::string replaced_data_length_string =
          std::to_string(result.replaced_data_size[i]);
      code += "__memcpy(" + ldram_to_string + ", " + cache_from_string + ", " +
              replaced_data_length_string + ", NRAM2LDRAM);\n";
    }
  }
  code += "__memcpy(" + cache_string + ", " + data_string + ", " +
          length_string + ", GDRAM2NRAM);\n";
  return cache_string;
}

std::string CudaLoadMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.load(cache_data);
  std::string cache_string =
      cache.name + "[" + std::to_string(result.cache_offset) + " + ";
  std::string data_string = data_name + "[" + std::to_string(data) + " + ";
  std::string length_string = std::to_string(length);
  switch (result.location) {
    case CacheHitLocation::CACHE:
      return cache_string;
    case CacheHitLocation::LDRAM:
      code += "TODO\n";
      return "";
    case CacheHitLocation::NOT_FOUND:
      code +=
          cache_string + "threadIdx.x] = " + data_string + "threadIdx.x];\n";
      return cache_string;
    default:
      return "";
  }
}

std::string CudaStoreMicro::generatorCode(Cache& cache, std::string& code) {
  return "";
}

std::string CudaAllocateMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.allocate(cache_data);
  // TODO mang thing
  std::string cache_string =
      cache.name + "[" + std::to_string(result.cache_offset) + " + ";
  return cache_string;
}

}  // namespace infini
