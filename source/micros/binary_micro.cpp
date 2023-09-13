#include "micros/binary_micro.h"
#include "micros/memory_micro.h"
#include "core/cache.h"

namespace infini {

std::string BangAddMicro::generatorCode(Cache& cache, std::string& code) {
  cache.lock();
  std::string left_cache =
      BangLoadMicro(left_name, left, length).generatorCode(cache, code);
  std::string right_cache =
      BangLoadMicro(right_name, right, length).generatorCode(cache, code);
  std::string output_cache =
      BangAllocateMicro(output_name, output, length).generatorCode(cache, code);
  code += "__bang_add(" + output_cache + ", " + left_cache + ", " +
          right_cache + ", " + std::to_string(length) + ");\n";
  cache.unlock();
  return "";
}

}  // namespace infini
