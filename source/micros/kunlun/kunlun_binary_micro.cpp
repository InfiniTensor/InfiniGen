#include "core/cache.h"
#include "core/utils.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"

namespace infini {
#define KUNLUN_BINARY_GENERATOR(OP, OP_STR, CAST)                              \
    std::string CAT(OP, Kunlun)::code(Cache &cache, std::string &code,         \
                                      int64_t indent) {                        \
        cache.lock();                                                          \
        std::string leftCache = LoadKunlun({left}).code(cache, code, indent);  \
        std::string rightCache =                                               \
            LoadKunlun({right}).code(cache, code, indent);                     \
        std::string outputCache =                                              \
            AllocateKunlun({output}).code(cache, code, indent);                \
        code += INDENTATION(indent) + "for (int i = 0; i < " +                 \
                std::to_string(length) + "; i++) {\n";                           \
        code += INDENTATION(indent + 1) + outputCache + "[i]" + " = ";         \
        code += leftCache + "[i] " + OP_STR + " " + rightCache + "[i];\n";     \
        code += INDENTATION(indent) + "}\n";                                   \
        cache.unlock();                                                        \
        return "";                                                             \
    }

KUNLUN_BINARY_GENERATOR(Add, "+", false)
KUNLUN_BINARY_GENERATOR(Sub, "-", false)
KUNLUN_BINARY_GENERATOR(Mul, "*", false)
KUNLUN_BINARY_GENERATOR(Div, "/", false)
KUNLUN_BINARY_GENERATOR(Eq, "==", true)
KUNLUN_BINARY_GENERATOR(Ge, ">=", true)
KUNLUN_BINARY_GENERATOR(Gt, ">", true)
KUNLUN_BINARY_GENERATOR(Le, "<=", true)
KUNLUN_BINARY_GENERATOR(Lt, "<", true)
KUNLUN_BINARY_GENERATOR(Ne, "!=", true)

REGISTER_MICRO(OperatorType::ADD, Platform::KUNLUN, AddKunlun::makeObj)
REGISTER_MICRO(OperatorType::SUB, Platform::KUNLUN, SubKunlun::makeObj)
REGISTER_MICRO(OperatorType::MUL, Platform::KUNLUN, MulKunlun::makeObj)
REGISTER_MICRO(OperatorType::DIV, Platform::KUNLUN, DivKunlun::makeObj)
REGISTER_MICRO(OperatorType::EQ, Platform::KUNLUN, EqKunlun::makeObj)
REGISTER_MICRO(OperatorType::GE, Platform::KUNLUN, GeKunlun::makeObj)
REGISTER_MICRO(OperatorType::GT, Platform::KUNLUN, GtKunlun::makeObj)
REGISTER_MICRO(OperatorType::LE, Platform::KUNLUN, LeKunlun::makeObj)
REGISTER_MICRO(OperatorType::LT, Platform::KUNLUN, LtKunlun::makeObj)
REGISTER_MICRO(OperatorType::NE, Platform::KUNLUN, NeKunlun::makeObj)

} // namespace infini
