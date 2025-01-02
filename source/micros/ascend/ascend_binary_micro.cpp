#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define ASCEND_BINARY_GENERATOR(OP, OP_STR, AUX_REQUIRED)                      \
    std::string CAT(OP, Ascend)::code(Cache &cache, std::string &code,         \
                                      int64_t indent) {                        \
        cache.lock();                                                          \
        std::string leftCache = LoadAscend({left}).code(cache, code, indent);  \
        std::string rightCache =                                               \
            LoadAscend({right}).code(cache, code, indent);                     \
        std::string outputCache =                                              \
            AllocateAscend({output}).code(cache, code, indent);                \
        if (AUX_REQUIRED) {                                                    \
            auto aux1 = new Tile(*output);                                     \
            std::string aux1Cache =                                            \
                AllocateAscend({aux1}).code(cache, code, indent);              \
            code += INDENTATION(indent) + std::string(OP_STR) + "(" +          \
                    outputCache + ", " + leftCache + ", " + rightCache +       \
                    ", " + aux1Cache + ", " + std::to_string(length) + ");\n"; \
            cache.unlock();                                                    \
            FreeAscend({aux1}).code(cache, code, indent);                      \
        } else {                                                               \
            code += INDENTATION(indent) + std::string(OP_STR) + "(" +          \
                    outputCache + ", " + leftCache + ", " + rightCache +       \
                    ", " + std::to_string(length) + ");\n";                    \
            cache.unlock();                                                    \
        }                                                                      \
        return "";                                                             \
    }

#define ASCEND_COMPARE_GENERATOR(OP, CMPMODE)                                  \
    std::string CAT(OP, Ascend)::code(Cache &cache, std::string &code,         \
                                      int64_t indent) {                        \
        cache.lock();                                                          \
        std::string leftCache = LoadAscend({left}).code(cache, code, indent);  \
        std::string rightCache =                                               \
            LoadAscend({right}).code(cache, code, indent);                     \
        std::string outputCache =                                              \
            AllocateAscend({output}).code(cache, code, indent);                \
        code += INDENTATION(indent) + "Compare(" + outputCache + ", " +        \
                leftCache + ", " + rightCache + ", CMPMODE::" + CMPMODE +      \
                ", " + std::to_string(length) + ");\n";                        \
        cache.unlock();                                                        \
        return "";                                                             \
    }

ASCEND_BINARY_GENERATOR(Add, "Add", false);
ASCEND_BINARY_GENERATOR(Sub, "Sub", false);
ASCEND_BINARY_GENERATOR(Mul, "Mul", false);
ASCEND_BINARY_GENERATOR(Div, "Div", false);
ASCEND_BINARY_GENERATOR(And, "And", false);
ASCEND_BINARY_GENERATOR(Or, "Or", false);
ASCEND_BINARY_GENERATOR(Xor, "Xor", true);
ASCEND_COMPARE_GENERATOR(Eq, "EQ");
ASCEND_COMPARE_GENERATOR(Ge, "GE");
ASCEND_COMPARE_GENERATOR(Gt, "GT");
ASCEND_COMPARE_GENERATOR(Le, "LE");
ASCEND_COMPARE_GENERATOR(Lt, "LT");
ASCEND_COMPARE_GENERATOR(Ne, "NE");

REGISTER_MICRO(OperatorType::ADD, Platform::ASCEND, AddAscend::makeObj)
REGISTER_MICRO(OperatorType::SUB, Platform::ASCEND, SubAscend::makeObj)
REGISTER_MICRO(OperatorType::MUL, Platform::ASCEND, MulAscend::makeObj)
REGISTER_MICRO(OperatorType::DIV, Platform::ASCEND, DivAscend::makeObj)
REGISTER_MICRO(OperatorType::EQ, Platform::ASCEND, EqAscend::makeObj)
REGISTER_MICRO(OperatorType::GE, Platform::ASCEND, GeAscend::makeObj)
REGISTER_MICRO(OperatorType::GT, Platform::ASCEND, GtAscend::makeObj)
REGISTER_MICRO(OperatorType::LE, Platform::ASCEND, LeAscend::makeObj)
REGISTER_MICRO(OperatorType::LT, Platform::ASCEND, LtAscend::makeObj)
REGISTER_MICRO(OperatorType::NE, Platform::ASCEND, NeAscend::makeObj)
REGISTER_MICRO(OperatorType::AND, Platform::ASCEND, AndAscend::makeObj)
REGISTER_MICRO(OperatorType::OR, Platform::ASCEND, OrAscend::makeObj)
REGISTER_MICRO(OperatorType::XOR, Platform::ASCEND, XorAscend::makeObj)

#undef ASCEND_BINARY_GENERATOR
#undef ASCEND_COMPARE_GENERATOR
} // namespace infini