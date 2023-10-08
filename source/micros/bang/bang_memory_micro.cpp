#include "core/utils.h"
#include "core/cache.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadBang::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  int64_t length_in_bytes = length * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.load(cache_data);
  std::string length_string = std::to_string(length_in_bytes);
  std::string cache_string = "(" + datatype_string(data_type) + " *)(" +
                             cache.name + " + " +
                             std::to_string(result.cache_offset) + ")";
  std::string data_string = name + " + " + std::to_string(offset);
  data_string += " + " + platform.taskIdx() + " * " + std::to_string(length);
  std::string ldram_from_string =
      cache.name + "_ldram + " + std::to_string(result.ldram_from_offset);

  if (result.location == CacheHitLocation::CACHE) {
    return cache_string;
  } else {
    for (int i = 0; i < result.ldram_to_offset.size(); i++) {
      std::string cache_from_string =
          cache.name + " + " +
          std::to_string(result.replaced_data_cache_offset[i]);
      std::string ldram_to_string =
          cache.name + "_ldram + " + std::to_string(result.ldram_to_offset[i]);
      std::string replaced_data_length_string =
          std::to_string(result.replaced_data_size[i]);
      code += indentation(indent) + "__memcpy(" + ldram_to_string + ", " +
              cache_from_string + ", " + replaced_data_length_string +
              ", NRAM2LDRAM);\n";
    }

    if (result.location == CacheHitLocation::LDRAM) {
      code += indentation(indent) + "__memcpy(" + cache_string + ", " +
              ldram_from_string + ", " + length_string + ", LDRAM2NRAM);\n";
      return cache_string;
    } else if (result.location == CacheHitLocation::NOT_FOUND) {
      code += indentation(indent) + "__memcpy(" + cache_string + ", " +
              data_string + ", " + length_string + ", GDRAM2NRAM);\n";
      return cache_string;
    } else {
      return "";
    }
  }
}

std::string StoreBang::generatorCode(Cache& cache, std::string& code,
                                     int64_t indent) {
  int64_t length_in_bytes = length * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.find(cache_data);
  std::string length_string = std::to_string(length_in_bytes);
  std::string cache_string = "(" + datatype_string(data_type) + " *)(" +
                             cache.name + " + " +
                             std::to_string(result.cache_offset) + ")";
  std::string data_string = name + " + " + std::to_string(offset);
  data_string += " + " + platform.taskIdx() + " * " + std::to_string(length);
  std::string ldram_from_string =
      cache.name + "_ldram + " + std::to_string(result.ldram_from_offset);

  if (result.location == CacheHitLocation::CACHE) {
    code += indentation(indent) + "__memcpy(" + data_string + ", " +
            cache_string + ", " + length_string + ", NRAM2GDRAM);\n";
    return cache_string;
  } else if (result.location == CacheHitLocation::LDRAM) {
    code += indentation(indent) + "__memcpy(" + data_string + ", " +
            ldram_from_string + ", " + length_string + ", LDRAM2GDRAM);\n";
    return ldram_from_string;
  } else {
    return "";
  }
}

std::string FreeBang::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  int64_t length_in_bytes = length * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.free(cache_data);
  return "";
}

std::string AllocateBang::generatorCode(Cache& cache, std::string& code,
                                        int64_t indent) {
  int64_t length_in_bytes = length * datatype_size(data_type);
  CacheData cache_data = CacheData(name, offset, length_in_bytes);
  auto result = cache.allocate(cache_data);
  std::string length_string = std::to_string(length_in_bytes);
  std::string cache_string = "(" + datatype_string(data_type) + " *)(" +
                             cache.name + " + " +
                             std::to_string(result.cache_offset) + ")";
  std::string data_string = name + " + " + std::to_string(offset);
  data_string += " + " + platform.taskIdx() + " * " + std::to_string(length);
  std::string ldram_from_string =
      cache.name + "_ldram + " + std::to_string(result.ldram_from_offset);

  for (int i = 0; i < result.ldram_to_offset.size(); i++) {
    std::string cache_from_string =
        cache.name + " + " +
        std::to_string(result.replaced_data_cache_offset[i]);
    std::string ldram_to_string =
        cache.name + "_ldram + " + std::to_string(result.ldram_to_offset[i]);
    std::string replaced_data_length_string =
        std::to_string(result.replaced_data_size[i]);
    code += indentation(indent) + "__memcpy(" + ldram_to_string + ", " +
            cache_from_string + ", " + replaced_data_length_string +
            ", NRAM2LDRAM);\n";
  }
  return cache_string;
}

}  // namespace infini