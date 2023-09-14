#include "micros/memory_micro.h"
#include "core/cache.h"

namespace infini {

std::string BangLoadMicro::generatorCode(Cache& cache, std::string& code,
                                         std::string coreIndex) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.load(cache_data);
  std::string length_string = std::to_string(length);
  std::string cache_string =
      cache.name + " + " + std::to_string(result.cache_offset);
  std::string data_string = data_name + " + " + std::to_string(data);
  data_string +=
      (coreIndex == "" ? "" : " + " + coreIndex + " * " + length_string);
  switch (result.location) {
    case CacheHitLocation::CACHE:
      return cache_string;
    case CacheHitLocation::LDRAM:
      code += "TODO\n";
      return "";
    case CacheHitLocation::NOT_FOUND:
      code += "__memcpy(" + cache_string + ", " + data_string + ", " +
              length_string + ", GDRAM2NRAM);\n";
      return cache_string;
    default:
      return "";
  }
}

std::string BangStoreMicro::generatorCode(Cache& cache, std::string& code,
                                          std::string coreIndex) {
  return "";
}

std::string BangAllocateMicro::generatorCode(Cache& cache, std::string& code,
                                             std::string coreIndex) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.allocate(cache_data);
  // TODO mang thing
  std::string cache_string =
      cache.name + " + " + std::to_string(result.cache_offset);
  return cache_string;
}

std::string CudaLoadMicro::generatorCode(Cache& cache, std::string& code,
                                         std::string coreIndex) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.load(cache_data);
  std::string length_string = std::to_string(length);
  std::string cache_string =
      cache.name + "[" + std::to_string(result.cache_offset) + " + ";
  std::string data_string = data_name + "[" + std::to_string(data) + " + ";
  data_string +=
      (coreIndex == "" ? "" : coreIndex + " * " + length_string + " + ");
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

std::string CudaStoreMicro::generatorCode(Cache& cache, std::string& code,
                                          std::string coreIndex) {
  return "";
}

std::string CudaAllocateMicro::generatorCode(Cache& cache, std::string& code,
                                             std::string coreIndex) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.allocate(cache_data);
  // TODO mang thing
  std::string cache_string =
      cache.name + "[" + std::to_string(result.cache_offset) + " + ";
  return cache_string;
}

}  // namespace infini
