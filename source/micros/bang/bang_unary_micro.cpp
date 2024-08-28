#include "core/cache.h"
#include "core/utils.h"
#include "micros/memory_micro.h"
#include "micros/unary_micro.h"

namespace infini {

#define BANG_UNARY_GENERATOR(OP, OP_STRING, AUX_REQUIRED)                      \
    std::string CAT(OP, Bang)::code(Cache &cache, std::string &code,           \
                                    int64_t indent) {                          \
        cache.lock();                                                          \
        std::string inputCache = LoadBang({input}).code(cache, code, indent);  \
        std::string outputCache =                                              \
            AllocateBang({output}).code(cache, code, indent);                  \
        if (AUX_REQUIRED) {                                                    \
            auto aux1 = new Tile(*output);                                     \
            auto aux2 = new Tile(*output);                                     \
            std::string aux1Cache =                                            \
                AllocateBang({aux1}).code(cache, code, indent);                \
            std::string aux2Cache =                                            \
                AllocateBang({aux2}).code(cache, code, indent);                \
            code += INDENTATION(indent) + std::string(OP_STRING) + "(" +       \
                    outputCache + ", " + inputCache + ", " + aux1Cache +       \
                    ", " + aux2Cache + ", " + std::to_string(length) + ");\n"; \
            cache.unlock();                                                    \
            FreeBang({aux1}).code(cache, code, indent);                        \
            FreeBang({aux2}).code(cache, code, indent);                        \
        } else {                                                               \
            code += INDENTATION(indent) + std::string(OP_STRING) + "(" +       \
                    outputCache + ", " + inputCache + ", " +                   \
                    std::to_string(length) + ");\n";                           \
            cache.unlock();                                                    \
        }                                                                      \
        return "";                                                             \
    }

BANG_UNARY_GENERATOR(Sqrt, "__bang_active_sqrthp", false)
BANG_UNARY_GENERATOR(Sigmoid, "__bang_taylor4_sigmoid", true)
BANG_UNARY_GENERATOR(Relu, "__bang_active_relu", false)
BANG_UNARY_GENERATOR(RSqrt, "__bang_active_rsqrthp", false)
BANG_UNARY_GENERATOR(Recip, "__bang_active_reciphp", false)
BANG_UNARY_GENERATOR(Cos, "__bang_taylor4_cos", true)
BANG_UNARY_GENERATOR(Sin, "__bang_taylor4_sin", true)
BANG_UNARY_GENERATOR(Tanh, "__bang_taylor4_tanh", true)

REGISTER_MICRO(OperatorType::SQRT, Platform::BANG, SqrtBang::makeObj)
REGISTER_MICRO(OperatorType::SIGMOID, Platform::BANG, SigmoidBang::makeObj)
REGISTER_MICRO(OperatorType::RELU, Platform::BANG, ReluBang::makeObj)
REGISTER_MICRO(OperatorType::RSQRT, Platform::BANG, RSqrtBang::makeObj)
REGISTER_MICRO(OperatorType::RECIP, Platform::BANG, RecipBang::makeObj)
REGISTER_MICRO(OperatorType::COS, Platform::BANG, CosBang::makeObj)
REGISTER_MICRO(OperatorType::SIN, Platform::BANG, SinBang::makeObj)
REGISTER_MICRO(OperatorType::TANH, Platform::BANG, TanhBang::makeObj)

#undef BANG_UNARY_GENERATOR

} // namespace infini
