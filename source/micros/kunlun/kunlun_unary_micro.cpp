#include "core/cache.h"
#include "core/utils.h"
#include "micros/memory_micro.h"
#include "micros/unary_micro.h"
namespace infini {

#define UNARY_LAMBDA(OP_STR, DTYPE, LAMBDA)                                    \
    "auto " + std::string(OP_STR) + " = [] __device__ (" +                     \
        dataTypeStr(DTYPE) + " in) -> " + dataTypeStr(DTYPE) + " {return " +   \
        std::string(LAMBDA) + ";};"

std::string kunlun_unary_kernel(std::string kernelName, TensorDataType dtype) {
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

#define KUNLUN_UNARY_GENERATOR(OP, OP_STR)                                     \
    std::string CAT(OP, Kunlun)::code(Cache &cache, std::string &code,         \
                                      int64_t indent) {                        \
        cache.lock();                                                          \
        code += INDENTATION(indent) + kunlun_unary_kernel(OP_STR, dataType) +  \
                ";\n";                                                         \
        std::string inputCache =                                               \
            LoadKunlun({input}).code(cache, code, indent);                     \
        std::string outputCache =                                              \
            AllocateKunlun({output}).code(cache, code, indent);                \
        code += INDENTATION(indent) + "for (int i = 0; i < " +                 \
                std::to_string(length) + "; i++) {\n";                         \
        code += INDENTATION(indent + 1) + outputCache +                        \
                "[i] = " + std::string(OP_STR) + "(" + inputCache + "[i]);\n"; \
        code += INDENTATION(indent) + "}\n";                                   \
        cache.unlock();                                                        \
        return "";                                                             \
    }

KUNLUN_UNARY_GENERATOR(Sqrt, "sqrt")
KUNLUN_UNARY_GENERATOR(RSqrt, "rsqrt")
KUNLUN_UNARY_GENERATOR(Relu, "relu")
KUNLUN_UNARY_GENERATOR(Sigmoid, "sigmoid")
KUNLUN_UNARY_GENERATOR(Recip, "recip")
KUNLUN_UNARY_GENERATOR(Sin, "sin")
KUNLUN_UNARY_GENERATOR(Cos, "cos")
KUNLUN_UNARY_GENERATOR(Tanh, "tanh")

REGISTER_MICRO(OperatorType::SQRT, Platform::KUNLUN, SqrtKunlun::makeObj)
REGISTER_MICRO(OperatorType::RSQRT, Platform::KUNLUN, RSqrtKunlun::makeObj)
REGISTER_MICRO(OperatorType::RELU, Platform::KUNLUN, ReluKunlun::makeObj)
REGISTER_MICRO(OperatorType::SIGMOID, Platform::KUNLUN, SigmoidKunlun::makeObj)
REGISTER_MICRO(OperatorType::RECIP, Platform::KUNLUN, RecipKunlun::makeObj)
REGISTER_MICRO(OperatorType::SIN, Platform::KUNLUN, SinKunlun::makeObj)
REGISTER_MICRO(OperatorType::COS, Platform::KUNLUN, CosKunlun::makeObj)
REGISTER_MICRO(OperatorType::TANH, Platform::KUNLUN, TanhKunlun::makeObj)

#undef UNARY_LAMBDA
#undef KUNLUN_UNARY_GENERATOR

} // namespace infini