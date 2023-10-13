#include "core/cache.h"
#include "core/utils.h"
#include "micros/unary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define BANG_UNARY_GENERATOR(OP, OP_STRING)                                 \
  std::string CAT(OP, Bang)::generatorCode(Cache& cache, std::string& code, \
                                           int64_t indent) {                \
    cache.lock();                                                           \
    std::string input_cache =                                               \
        LoadBang(OperandType{input_name, input_offset, length, data_type})  \
            .generatorCode(cache, code, indent);                            \
    std::string output_cache =                                              \
        AllocateBang(                                                       \
            OperandType{output_name, output_offset, length, data_type})     \
            .generatorCode(cache, code, indent);                            \
    code += indentation(indent) + std::string(OP_STRING) + "(" +            \
            output_cache + ", " + input_cache + ", " +                      \
            std::to_string(length) + ");\n";                                \
    cache.unlock();                                                         \
    return "";                                                              \
  }

BANG_UNARY_GENERATOR(Sqrt, "__bang_active_sqrthp")
BANG_UNARY_GENERATOR(Sigmoid, "__bang_taylor4_sigmoid")
BANG_UNARY_GENERATOR(Relu, "__bang_active_relu")
BANG_UNARY_GENERATOR(RSqrt, "__bang_active_rsqrthp")
BANG_UNARY_GENERATOR(Recip, "__bang_active_reciphp")

REGISTER_MICRO(OperatorType::SQRT, Platform::BANG, SqrtBang::makeObj)
REGISTER_MICRO(OperatorType::SIGMOID, Platform::BANG, SigmoidBang::makeObj)
REGISTER_MICRO(OperatorType::RELU, Platform::BANG, ReluBang::makeObj)
REGISTER_MICRO(OperatorType::RSQRT, Platform::BANG, RSqrtBang::makeObj)
REGISTER_MICRO(OperatorType::RECIP, Platform::BANG, RecipBang::makeObj)

#undef BANG_UNARY_GENERATOR

}  // namespace infini
