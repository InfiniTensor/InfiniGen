#include "micros/binary_micro.h"
#include "core/cache.h"

namespace infini {

std::string BangAddMicro::generatorCode(Cache& cache) {
  std::string code;
  cache.lock();
  CacheData left_data = CacheData(left_name, left, length);
  CacheData right_data = CacheData(right_name, right, length);
  CacheData output_data = CacheData(output_name, output, length);
  auto result = cache.load(left_data);
  if (result.location == CacheHitLocation::NOT_FOUND) {
    code += "__bang_memcpy()\n";
  }
  result.printInformation();
  result = cache.load(right_data);
  if (result.location == CacheHitLocation::NOT_FOUND) {
    code += "__bang_memcpy()\n";
  }
  result.printInformation();
  result = cache.allocate(output_data);
  if (result.location == CacheHitLocation::NOT_FOUND) {
    code += "__bang_memcpy()\n";
  }
  result.printInformation();
  code += "bang_add()\n";
  cache.unlock();
  return code;
}

}  // namespace infini
