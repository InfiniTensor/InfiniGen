#include "micros/bang/bang_binary_micro.h"
#include "micros/bang/bang_memery_micro.h"

namespace infini {

#define CUDA_BINARY_IMPLEMENT(OP, OP_STRING)                                 \
  std::string OP##Cuda::generatorCode(Cache& cache, std::string& code,       \
                                      int64_t indent) {                      \
    cache.lock();                                                            \
    std::string left_cache =                                                 \
        LoadCuda(std::tuple(left_name, left_offset, length))                 \
            .generatorCode(cache, code, indent);                             \
    std::string right_cache =                                                \
        LoadCuda(std::tuple(right_name, right_offset, length))               \
            .generatorCode(cache, code, indent);                             \
    std::string output_cache =                                               \
        AllocateCuda(std::tuple(output_name, output_offset, length))         \
            .generatorCode(cache, code, indent);                             \
    code += indentation(indent) + output_cache + "] = " + left_cache + "] " + \
            std::string(OP_STRING) + " " + right_cache + "];\n";              \
    cache.unlock();                                                          \
    return "";                                                               \
  }

CUDA_BINARY_IMPLEMENT(Add, "+")
CUDA_BINARY_IMPLEMENT(Sub, "-")
CUDA_BINARY_IMPLEMENT(Mul, "*")

} // infini
