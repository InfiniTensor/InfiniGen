#include "core/cache.h"
#include "core/utils.h"
#include "micros/memory_micro.h"
#include "micros/unary_micro.h"

namespace infini {

#define UNARY_LAMBDA(OP_STR, DTYPE, LAMBDA)                                    \
    "auto " + std::string(OP_STR) + " = [] __device__ (" +                     \
        dataTypeStr(DTYPE) + " in) -> " + dataTypeStr(DTYPE) + " {return " +   \
        std::string(LAMBDA) + ";};"

std::string unary_kernel(std::string kernelName, TensorDataType dtype) {
    if (kernelName == "relu") {
        return UNARY_LAMBDA(kernelName, dtype, "in > 0 ? in : 0");
    } else if (kernelName == "sigmoid") {
        return UNARY_LAMBDA(kernelName, dtype, "1.0 / (1.0 + expf(-in))");
    } else if (kernelName == "recip") {
        return UNARY_LAMBDA(kernelName, dtype, "1.0 / in");
    } else {
        return "";
    }
}

#define CUDA_UNARY_GENERATOR(OP, OP_STR)                                       \
    std::string CAT(OP, Cuda)::code(Cache &cache, std::string &code,           \
                                    int64_t indent) {                          \
        cache.lock();                                                          \
        code += INDENTATION(indent) + unary_kernel(OP_STR, dataType) + "\n";   \
        std::string inputCache = LoadCuda({input}).code(cache, code, indent);  \
        std::string outputCache =                                              \
            AllocateCuda({output}).code(cache, code, indent);                  \
        code += INDENTATION(indent) + outputCache + " = " +                    \
                std::string(OP_STR) + "(" + inputCache + ");\n";               \
        cache.unlock();                                                        \
        return "";                                                             \
    }

// unpack defination
CUDA_UNARY_GENERATOR(Sqrt, "sqrt")
CUDA_UNARY_GENERATOR(RSqrt, "rsqrt")
CUDA_UNARY_GENERATOR(Relu, "relu")
CUDA_UNARY_GENERATOR(Sigmoid, "sigmoid")
CUDA_UNARY_GENERATOR(Recip, "recip")
CUDA_UNARY_GENERATOR(Sin, "sin")
CUDA_UNARY_GENERATOR(Cos, "cos")
CUDA_UNARY_GENERATOR(Tanh, "tanhf")

// register micros
REGISTER_MICRO(OperatorType::SQRT, Platform::CUDA, SqrtCuda::makeObj)
REGISTER_MICRO(OperatorType::RSQRT, Platform::CUDA, RSqrtCuda::makeObj)
REGISTER_MICRO(OperatorType::RELU, Platform::CUDA, ReluCuda::makeObj)
REGISTER_MICRO(OperatorType::SIGMOID, Platform::CUDA, SigmoidCuda::makeObj)
REGISTER_MICRO(OperatorType::RECIP, Platform::CUDA, RecipCuda::makeObj)
REGISTER_MICRO(OperatorType::SIN, Platform::CUDA, SinCuda::makeObj)
REGISTER_MICRO(OperatorType::COS, Platform::CUDA, CosCuda::makeObj)
REGISTER_MICRO(OperatorType::TANH, Platform::CUDA, TanhCuda::makeObj)

#undef CUDA_UNARY_GENERATOR
} // namespace infini
