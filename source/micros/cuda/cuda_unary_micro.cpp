#include "core/cache.h"
#include "core/utils.h"
#include "micros/unary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

std::string SqrtCuda::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  cache.lock();
  std::string input_cache =
      LoadCuda(OperandType{input_name, input_offset, length, data_type})
          .generatorCode(cache, code, indent);
  std::string output_cache =
      AllocateCuda(OperandType{output_name, output_offset, length, data_type})
          .generatorCode(cache, code, indent);
  code += indentation(indent) + output_cache + "] = " + "sqrt(" + input_cache +
          "]);\n";
  cache.unlock();
  return "";
}

std::string RSqrtCuda::generatorCode(Cache& cache, std::string& code,
                                     int64_t indent) {
  cache.lock();
  std::string input_cache =
      LoadCuda(OperandType{input_name, input_offset, length, data_type})
          .generatorCode(cache, code, indent);
  std::string output_cache =
      AllocateCuda(OperandType{output_name, output_offset, length, data_type})
          .generatorCode(cache, code, indent);
  code += indentation(indent) + output_cache + "] = " + "rsqrt(" + input_cache +
          "]);\n";
  cache.unlock();
  return "";
}

std::string SigmoidCuda::generatorCode(Cache& cache, std::string& code,
                                     int64_t indent) {
  cache.lock();
  std::string input_cache =
      LoadCuda(OperandType{input_name, input_offset, length, data_type})
          .generatorCode(cache, code, indent);
  std::string output_cache =
      AllocateCuda(OperandType{output_name, output_offset, length, data_type})
          .generatorCode(cache, code, indent);
  code += indentation(indent) + output_cache + "] = " + "sigmoid" + input_cache +
          "]);\n";
  cache.unlock();
  return "";
}


}  // namespace infini