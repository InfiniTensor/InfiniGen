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
      cache.name + "_ldram + " + std::to_string(result.ldram_from_offset);
  int64_t replaced_data_count = result.ldram_to_offset.size();

  if (result.location == CacheHitLocation::CACHE) {
    return cache_string;
  } else {
    if (replaced_data_count > 0) {
      for (int i = 0; i < replaced_data_count; i++) {
        std::string cache_from_string =
            cache.name + " + " +
            std::to_string(result.replaced_data_cache_offset[i]);
        std::string ldram_to_string = cache.name + "_ldram + " +
                                      std::to_string(result.ldram_to_offset[i]);
        std::string replaced_data_length_string =
            std::to_string(result.replaced_data_size[i]);
        code += "__memcpy(" + ldram_to_string + ", " + cache_from_string +
                ", " + replaced_data_length_string + ", NRAM2LDRAM);\n";
      }
    }
    if (result.location == CacheHitLocation::LDRAM) {
      code += "__memcpy(" + cache_string + ", " + ldram_from_string + ", " +
              length_string + ", LDRAM2NRAM);\n";
      return cache_string;
    } else if (result.location == CacheHitLocation::NOT_FOUND) {
      code += "__memcpy(" + cache_string + ", " + data_string + ", " +
              length_string + ", GDRAM2NRAM);\n";
      return cache_string;
    } else {
      return "";
    }
  }
}

std::string BangStoreMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.find(cache_data);
  std::string cache_string =
      cache.name + " + " + std::to_string(result.cache_offset);
  std::string data_string = data_name + " + " + std::to_string(data);
  std::string length_string = std::to_string(length);
  std::string ldram_from_string =
      cache.name + "_ldram + " + std::to_string(result.ldram_from_offset);

  if (result.location == CacheHitLocation::CACHE) {
    code += "__memcpy(" + data_string + ", " + cache_string + ", " +
            length_string + ", NRAM2GDRAM);\n";
    return cache_string;
  } else if (result.location == CacheHitLocation::LDRAM) {
    code += "__memcpy(" + data_string + ", " + ldram_from_string + ", " +
            length_string + ", LDRAM2GDRAM);\n";
    return ldram_from_string;
  } else {
    return "";
  }
}

std::string BangFreeMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.free(cache_data);
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
          cache.name + "_ldram + " + std::to_string(result.ldram_to_offset[i]);
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
  std::string ldram_from_string =
      cache.name + "[" + std::to_string(result.ldram_from_offset) + " + ";
  int64_t replaced_data_count = result.ldram_to_offset.size();

  if (result.location == CacheHitLocation::CACHE) {
    return cache_string;
  } else {
    if (replaced_data_count > 0) {
      for (int i = 0; i < replaced_data_count; i++) {
        std::string cache_from_string =
            cache.name + "[" +
            std::to_string(result.replaced_data_cache_offset[i]) + " + ";
        std::string ldram_to_string = cache.name + "_ldram + " +
                                      std::to_string(result.ldram_to_offset[i]);
        std::string replaced_data_length_string =
            std::to_string(result.replaced_data_size[i]);
        code += "if (threadIdx.x < " + replaced_data_length_string + " ) {\n" +
                "  " + ldram_to_string + "threadIdx.x] = " + cache_from_string +
                "threadIdx.x]\n" + "}\n";
      }
    }
    if (result.location == CacheHitLocation::LDRAM) {
      code += cache_string + "threadIdx.x] = " + ldram_from_string +
              "threadIdx.x];\n";
      return cache_string;
    } else if (result.location == CacheHitLocation::NOT_FOUND) {
      code +=
          cache_string + "threadIdx.x] = " + data_string + "threadIdx.x];\n";
      return cache_string;
    } else {
      return "";
    }
  }
}

std::string CudaStoreMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.find(cache_data);
  std::string cache_string =
      cache.name + "[" + std::to_string(result.cache_offset) + " + ";
  std::string data_string = data_name + "[" + std::to_string(data) + " + ";
  std::string length_string = std::to_string(length);
  std::string ldram_from_string =
      cache.name + "[" + std::to_string(result.ldram_from_offset) + " + ";

  if (result.location == CacheHitLocation::CACHE) {
    code += data_string + "threadIdx.x] = " + cache_string + "threadIdx.x];\n";
    return cache_string;
  } else if (result.location == CacheHitLocation::LDRAM) {
    code +=
        data_string + "threadIdx.x] = " + ldram_from_string + "threadIdx.x];\n";
    return ldram_from_string;
  } else {
    return "";
  }
}

std::string CudaAllocateMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.load(cache_data);
  std::string cache_string =
      cache.name + "[" + std::to_string(result.cache_offset) + " + ";
  std::string data_string = data_name + "[" + std::to_string(data) + " + ";
  std::string length_string = std::to_string(length);
  std::string ldram_from_string =
      cache.name + "[" + std::to_string(result.ldram_from_offset) + " + ";
  int64_t replaced_data_count = result.ldram_to_offset.size();
  if (replaced_data_count > 0) {
    for (int i = 0; i < replaced_data_count; i++) {
      std::string cache_from_string =
          cache.name + "[" +
          std::to_string(result.replaced_data_cache_offset[i]) + " + ";
      std::string ldram_to_string =
          cache.name + "_ldram + " + std::to_string(result.ldram_to_offset[i]);
      std::string replaced_data_length_string =
          std::to_string(result.replaced_data_size[i]);
      code += "if (threadIdx.x < " + replaced_data_length_string + " ) {\n" +
              "  " + ldram_to_string + "threadIdx.x] = " + cache_from_string +
              "threadIdx.x]\n" + "}\n";
    }
  }
  code += cache_string + "threadIdx.x] = " + data_string + "threadIdx.x];\n";
  return cache_string;
}

std::string CudaFreeMicro::generatorCode(Cache& cache, std::string& code) {
  CacheData cache_data = CacheData(data_name, data, length);
  auto result = cache.free(cache_data);
  return "";
}

}  // namespace infini
