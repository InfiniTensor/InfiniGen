#include "micros/binary_micro.h"
#include "micros/memory_micro.h"
#include "core/cache.h"
#include "core/utils.h"

namespace infini {
/**
 * Binary kernel include:
 *  - ADD
 *  - SUB
 *  - MUL
 */

/* BANG kernel Implementation */
#define BANG_BINARY_GENERATOR(OP, OP_STRING)                                  \
                                                                              \
  std::string Bang##OP##Micro::generatorCode(Cache& cache, std::string& code, \
                                             std::string coreIndex) {         \
    cache.lock();                                                             \
    std::string left_cache = BangLoadMicro(left_name, left, length)           \
                                 .generatorCode(cache, code, coreIndex);      \
    std::string right_cache = BangLoadMicro(right_name, right, length)        \
                                  .generatorCode(cache, code, coreIndex);     \
    std::string output_cache = BangAllocateMicro(output_name, output, length) \
                                   .generatorCode(cache, code, coreIndex);    \
    code += "__bang_" + std::string(OP_STRING) + "(" + output_cache + ", " +  \
            left_cache + ", " + right_cache + ", " + std::to_string(length) + \
            ");\n";                                                           \
    cache.unlock();                                                           \
    return "";                                                                \
  }

/* CUDA kernel implementation */
#define CUDA_BINARY_GENERATOR(OP, OP_STRING)                                  \
  std::string Cuda##OP##Micro::generatorCode(Cache& cache, std::string& code, \
                                             std::string coreIndex) {         \
    cache.lock();                                                             \
    std::string left_cache = CudaLoadMicro(left_name, left, length)           \
                                 .generatorCode(cache, code, coreIndex);      \
    std::string right_cache = CudaLoadMicro(right_name, right, length)        \
                                  .generatorCode(cache, code, coreIndex);     \
    std::string output_cache = CudaAllocateMicro(output_name, output, length) \
                                   .generatorCode(cache, code, coreIndex);    \
    code += output_cache + "] = " + left_cache + "]" +                        \
            std::string(OP_STRING) + right_cache + "];\n";                    \
    cache.unlock();                                                           \
    return "";                                                                \
  }

// BANG
BANG_BINARY_GENERATOR(Add, "add")
BANG_BINARY_GENERATOR(Sub, "sub")
BANG_BINARY_GENERATOR(Mul, "mul")

// CUDA
CUDA_BINARY_GENERATOR(Add, "+")
CUDA_BINARY_GENERATOR(Sub, "-")
CUDA_BINARY_GENERATOR(Mul, "*")

#undef BANG_BINARY_GENERATOR
#undef CUDA_BINARY_GENERATOR

}  // namespace infini
