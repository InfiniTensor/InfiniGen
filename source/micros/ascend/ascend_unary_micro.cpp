#include "core/cache.h"
#include "core/utils.h"
#include "micros/memory_micro.h"
#include "micros/unary_micro.h"

namespace infini {

#define ASCEND_UNARY_GENERATOR(OP, OP_STR, AUX_REQUIRED)                       \
    std::string CAT(OP, Ascend)::code(Cache &cache, std::string &code,         \
                                      int64_t indent) {                        \
        cache.lock();                                                          \
        std::string inputCache =                                               \
            LoadAscend({input}).code(cache, code, indent);                     \
        std::string outputCache =                                              \
            AllocateAscend({output}).code(cache, code, indent);                \
        if (AUX_REQUIRED) {                                                    \
            auto aux = new Tile(*output);                                      \
            std::string auxCache =                                             \
                AllocateAscend({aux}).code(cache, code, indent);               \
            code += INDENTATION(indent) + std::string(OP_STR) + "(" +          \
                    outputCache + ", " + inputCache + ", " + auxCache + ", " + \
                    std::to_string(length) + ");\n";                           \
            cache.unlock();                                                    \
            FreeAscend({aux}).code(cache, code, indent);                       \
        } else {                                                               \
            code += INDENTATION(indent) + std::string(OP_STR) + "(" +          \
                    outputCache + ", " + inputCache + ", " +                   \
                    std::to_string(length) + ");\n";                           \
            cache.unlock();                                                    \
        }                                                                      \
        return "";                                                             \
    }

ASCEND_UNARY_GENERATOR(Sqrt, "Sqrt", false);
ASCEND_UNARY_GENERATOR(Sigmoid, "Sigmoid", true);
ASCEND_UNARY_GENERATOR(Relu, "Relu", false);
ASCEND_UNARY_GENERATOR(RSqrt, "RSqrt", false);
ASCEND_UNARY_GENERATOR(Recip, "Reciprocal", false);
ASCEND_UNARY_GENERATOR(Cos, "Cos", true);
ASCEND_UNARY_GENERATOR(Sin, "Sin", true);
ASCEND_UNARY_GENERATOR(Tanh, "Tanh", true);

REGISTER_MICRO(OperatorType::SQRT, Platform::ASCEND, SqrtAscend::makeObj)
REGISTER_MICRO(OperatorType::SIGMOID, Platform::ASCEND, SigmoidAscend::makeObj)
REGISTER_MICRO(OperatorType::RELU, Platform::ASCEND, ReluAscend::makeObj)
REGISTER_MICRO(OperatorType::RSQRT, Platform::ASCEND, RSqrtAscend::makeObj)
REGISTER_MICRO(OperatorType::RECIP, Platform::ASCEND, RecipAscend::makeObj)
REGISTER_MICRO(OperatorType::COS, Platform::ASCEND, CosAscend::makeObj)
REGISTER_MICRO(OperatorType::SIN, Platform::ASCEND, SinAscend::makeObj)
REGISTER_MICRO(OperatorType::TANH, Platform::ASCEND, TanhAscend::makeObj)

#undef ASCEND_UNARY_GENERATOR
} // namespace infini