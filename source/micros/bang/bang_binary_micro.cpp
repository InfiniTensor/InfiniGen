#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define BANG_BINARY_GENERATOR(OP, OP_STR)                                      \
    std::string CAT(OP, Bang)::code(Cache &cache, std::string &code,           \
                                    int64_t indent) {                          \
        std::string leftCache = LoadBang({left}).code(cache, code, indent);    \
        std::string rightCache = LoadBang({right}).code(cache, code, indent);  \
        std::string outputCache =                                              \
            AllocateBang({output}).code(cache, code, indent);                  \
        code += INDENTATION(indent) + "__bang_" + std::string(OP_STR) + "(" +  \
                outputCache + ", " + leftCache + ", " + rightCache + ", " +    \
                std::to_string(length) + ");\n";                               \
        return "";                                                             \
    }

BANG_BINARY_GENERATOR(Add, "add")
BANG_BINARY_GENERATOR(Sub, "sub")
BANG_BINARY_GENERATOR(Mul, "mul")
BANG_BINARY_GENERATOR(Div, "div")
BANG_BINARY_GENERATOR(Eq, "eq")
BANG_BINARY_GENERATOR(Ge, "ge")
BANG_BINARY_GENERATOR(Gt, "gt")
BANG_BINARY_GENERATOR(Le, "le")
BANG_BINARY_GENERATOR(Lt, "lt")
BANG_BINARY_GENERATOR(Ne, "ne")
BANG_BINARY_GENERATOR(And, "and")
BANG_BINARY_GENERATOR(Or, "or")
BANG_BINARY_GENERATOR(Xor, "xor")

// // Div
// std::string DivBang::code(Cache &cache, std::string &code, int64_t indent) {
//     std::string leftCache = LoadBang({left}).code(cache, code, indent);
//     std::string rightCache = LoadBang({right}).code(cache, code, indent);
//     std::string outputCache = AllocateBang({output}).code(cache, code,
//     indent); auto recip = new Tile(*right); std::string recipCache =
//     AllocateBang({recip}).code(cache, code, indent); code +=
//     INDENTATION(indent) + "__bang_active_reciphp(" + recipCache + ", " +
//             rightCache + ", " + std::to_string(length) + ");\n";
//     code += INDENTATION(indent) + "__bang_mul(" + outputCache + ", " +
//             leftCache + ", " + recipCache + ", " + std::to_string(length) +
//             ");\n";
//     FreeBang({recip}).code(cache, code, indent);
//     delete recip;
//     return "";
// }

/**
 * Register Micros
 */
// BANG
REGISTER_MICRO(OperatorType::ADD, Platform::BANG, AddBang::makeObj)
REGISTER_MICRO(OperatorType::SUB, Platform::BANG, SubBang::makeObj)
REGISTER_MICRO(OperatorType::MUL, Platform::BANG, MulBang::makeObj)
REGISTER_MICRO(OperatorType::DIV, Platform::BANG, DivBang::makeObj)
REGISTER_MICRO(OperatorType::EQ, Platform::BANG, EqBang::makeObj)
REGISTER_MICRO(OperatorType::GE, Platform::BANG, GeBang::makeObj)
REGISTER_MICRO(OperatorType::GT, Platform::BANG, GtBang::makeObj)
REGISTER_MICRO(OperatorType::LE, Platform::BANG, LeBang::makeObj)
REGISTER_MICRO(OperatorType::LT, Platform::BANG, LtBang::makeObj)
REGISTER_MICRO(OperatorType::NE, Platform::BANG, NeBang::makeObj)
REGISTER_MICRO(OperatorType::AND, Platform::BANG, AndBang::makeObj)
REGISTER_MICRO(OperatorType::OR, Platform::BANG, OrBang::makeObj)
REGISTER_MICRO(OperatorType::XOR, Platform::BANG, XorBang::makeObj)
// REGISTER_MICRO(OperatorType::FLOORMOD, Platform::BANG, FloorModBang::makeObj)
// REGISTER_MICRO(OperatorType::FLOORDIV, Platform::BANG, FloorDivBang::makeObj)

#undef BANG_BINARY_GENERATOR

} // namespace infini
