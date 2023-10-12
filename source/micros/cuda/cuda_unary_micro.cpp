#include "core/cache.h"
#include "core/utils.h"
#include "micros/unary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

std::map<std::string, std::string> unary_kernels = {
    {"relu", "[] __device__ (float in) -> float {return in > 0 ? in : 0; }"},
    {"sigmoid",
     "[] __device__ (float in) -> float {return 1.0 / (1.0 + expf(-in)); }"},
    {"recip", "[] __device__ (float in) -> float {return 1.0 / in; }"}};

#define CUDA_UNARY_GENERATOR(OP, OP_STR)                                     \
  std::string CAT(OP, Cuda)::generatorCode(Cache& cache, std::string& code,  \
                                           int64_t indent) {                 \
    cache.lock();                                                            \
    if (unary_kernels.count(OP_STR) > 0) {                                   \
      code += "auto " + std::string(OP_STR) + " = " + unary_kernels[OP_STR]; \
    }                                                                        \
    std::string input_cache =                                                \
        LoadCuda(OperandType{input_name, input_offset, length, data_type})   \
            .generatorCode(cache, code, indent);                             \
    std::string output_cache =                                               \
        AllocateCuda(                                                        \
            OperandType{output_name, output_offset, length, data_type})      \
            .generatorCode(cache, code, indent);                             \
    code += indentation(indent) + output_cache +                             \
            "] = " + std::string(OP_STR) + "(" + input_cache + "]);\n";      \
    cache.unlock();                                                          \
    return "";                                                               \
  }

// unpack defination
CUDA_UNARY_GENERATOR(Sqrt, "sqrt")
CUDA_UNARY_GENERATOR(RSqrt, "rsqrt")
CUDA_UNARY_GENERATOR(Relu, "relu")
CUDA_UNARY_GENERATOR(Sigmoid, "sigmoid")
CUDA_UNARY_GENERATOR(Recip, "recip")

// register micros
REGISTER_MICRO(OperatorType::SQRT, Platform::CUDA, SqrtCuda::makeObj)
REGISTER_MICRO(OperatorType::RSQRT, Platform::CUDA, RSqrtCuda::makeObj)
REGISTER_MICRO(OperatorType::RELU, Platform::CUDA, ReluCuda::makeObj)
REGISTER_MICRO(OperatorType::SIGMOID, Platform::CUDA, SigmoidCuda::makeObj)
REGISTER_MICRO(OperatorType::RECIP, Platform::CUDA, RecipCuda::makeObj)

}  // namespace infini
