#include "core/utils.h"
#include "core/cache.h"
#include "micros/memory_micro.h"
#include "micros/binary_micro.h"

namespace infini {

#define CUDA_GENERATOR(OP, OP_STR)                                            \
  std::string CAT(OP, Cuda)::generatorCode(Cache& cache, std::string& code,   \
                                           int64_t indent) {                  \
    cache.lock();                                                             \
    std::string left_cache =                                                  \
        LoadCuda(OperandType{left_name, left_offset, length, data_type})      \
            .generatorCode(cache, code, indent);                              \
    std::string right_cache =                                                 \
        LoadCuda(OperandType{right_name, right_offset, length, data_type})    \
            .generatorCode(cache, code, indent);                              \
    std::string output_cache =                                                \
        AllocateCuda(                                                         \
            OperandType{output_name, output_offset, length, data_type})       \
            .generatorCode(cache, code, indent);                              \
    code += indentation(indent) + output_cache + "] = " + left_cache + "] " + \
            std::string(OP_STR) + " " + right_cache + "];\n";                 \
    cache.unlock();                                                           \
    return "";                                                                \
  }

CUDA_GENERATOR(Add, "+")
CUDA_GENERATOR(Sub, "-")
CUDA_GENERATOR(Mul, "*")

#undef CUDA_GENERATOR

}  // namespace infini
