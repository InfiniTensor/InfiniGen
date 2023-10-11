#include "core/cache.h"
#include "core/utils.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define BANG_GENERATOR(OP, OP_STR)                                          \
  std::string CAT(OP, Bang)::generatorCode(Cache& cache, std::string& code, \
                                           int64_t indent) {                \
    cache.lock();                                                           \
    std::string left_cache =                                                \
        LoadBang(OperandType{left_name, left_offset, length, data_type})    \
            .generatorCode(cache, code, indent);                            \
    std::string right_cache =                                               \
        LoadBang(OperandType{right_name, right_offset, length, data_type})  \
            .generatorCode(cache, code, indent);                            \
    std::string output_cache =                                              \
        AllocateBang(                                                       \
            OperandType{output_name, output_offset, length, data_type})     \
            .generatorCode(cache, code, indent);                            \
    code += indentation(indent) + "__bang_" + std::string(OP_STR) + "(" +   \
            output_cache + ", " + left_cache + ", " + right_cache + ", " +  \
            std::to_string(length) + ");\n";                                \
    cache.unlock();                                                         \
    return "";                                                              \
  }

BANG_GENERATOR(Add, "add")
BANG_GENERATOR(Sub, "sub")
BANG_GENERATOR(Mul, "mul")
// BANG_GENERATOR(Div, "div")
BANG_GENERATOR(Eq, "eq")
BANG_GENERATOR(Ge, "ge")
BANG_GENERATOR(Gt, "gt")
BANG_GENERATOR(Le, "le")
BANG_GENERATOR(Lt, "lt")
BANG_GENERATOR(Ne, "ne")
BANG_GENERATOR(And, "and")
BANG_GENERATOR(Or, "or")
BANG_GENERATOR(Xor, "xor")

/**
 * Register Micros
 */
// BANG
REGISTER_MICRO(OperatorType::ADD, Platform::BANG, AddBang::makeObj)
REGISTER_MICRO(OperatorType::SUB, Platform::BANG, SubBang::makeObj)
REGISTER_MICRO(OperatorType::MUL, Platform::BANG, MulBang::makeObj)
// REGISTER_MICRO(OperatorType::DIV, Platform::BANG, DivBang::makeObj)
REGISTER_MICRO(OperatorType::EQ, Platform::BANG, EqBang::makeObj)
REGISTER_MICRO(OperatorType::GE, Platform::BANG, GeBang::makeObj)
REGISTER_MICRO(OperatorType::GT, Platform::BANG, GtBang::makeObj)
REGISTER_MICRO(OperatorType::LE, Platform::BANG, LeBang::makeObj)
REGISTER_MICRO(OperatorType::LT, Platform::BANG, LtBang::makeObj)
REGISTER_MICRO(OperatorType::NE, Platform::BANG, NeBang::makeObj)
REGISTER_MICRO(OperatorType::AND, Platform::BANG, AndBang::makeObj)
REGISTER_MICRO(OperatorType::OR, Platform::BANG, OrBang::makeObj)
REGISTER_MICRO(OperatorType::XOR, Platform::BANG, XorBang::makeObj)

#undef BANG_GENERATOR

}  // namespace infini
