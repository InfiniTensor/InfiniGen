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

std::string CudaAddMicro::generatorCode(Cache& cache, std::string& code) {
  cache.lock();
  code += "int index = threadIdx.x;\n";
  std::string left_cache =
      CudaLoadMicro(left_name, left, length).generatorCode(cache, code);
  std::string right_cache =
      CudaLoadMicro(right_name, right, length).generatorCode(cache, code);
  std::string output_cache =
      CudaAllocateMicro(output_name, output, length).generatorCode(cache, code);
  code += output_cache + "index] = " + left_cache + "index]" + " + " +
          right_cache + "idnex];\n";
  cache.unlock();
  return "";
}

}  // namespace infini
