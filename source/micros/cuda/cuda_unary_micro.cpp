#include "core/cache.h"
#include "core/utils.h"
#include "micros/unary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define UNARY_LAMBDA(OP_STR, DTYPE, LAMBDA)                          \
  "auto " + std::string(OP_STR) + " = [] __device__ (" +             \
      datatype_string(DTYPE) + " in) -> " + datatype_string(DTYPE) + \
      " {return " + std::string(LAMBDA) + ";}"

std::string unary_kernel(std::string kernel_name, TensorDatatype dtype) {
  if (kernel_name == "relu") {
    return UNARY_LAMBDA(kernel_name, dtype, "in > 0 ? in : 0");
  } else if (kernel_name == "sigmoid") {
    return UNARY_LAMBDA(kernel_name, dtype, "1.0 / (1.0 + expf(-in));");
  } else if (kernel_name == "recip") {
    return UNARY_LAMBDA(kernel_name, dtype, "1.0 / in");
  } else {
    return "";
  }
}

#define CUDA_UNARY_GENERATOR(OP, OP_STR)                                    \
  std::string CAT(OP, Cuda)::generatorCode(Cache& cache, std::string& code, \
                                           int64_t indent) {                \
    cache.lock();                                                           \
    code += unary_kernel(OP_STR, data_type);                                \
    std::string input_cache =                                               \
        LoadCuda(OperandType{input_name, input_offset, length, data_type})  \
            .generatorCode(cache, code, indent);                            \
    std::string output_cache =                                              \
        AllocateCuda(                                                       \
            OperandType{output_name, output_offset, length, data_type})     \
            .generatorCode(cache, code, indent);                            \
    code += indentation(indent) + output_cache +                            \
            "] = " + std::string(OP_STR) + "(" + input_cache + "]);\n";     \
    cache.unlock();                                                         \
    return "";                                                              \
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

#undef CUDA_UNARY_GENERATOR
}  // namespace infini
