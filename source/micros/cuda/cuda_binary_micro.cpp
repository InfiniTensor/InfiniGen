#include "core/cache.h"
#include "core/utils.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define CUDA_BINARY_GENERATOR(OP, OP_STR, CAST)                                \
    std::string CAT(OP, Cuda)::code(Cache &cache, std::string &code,           \
                                    int64_t indent) {                          \
        std::string leftCache = LoadCuda({left}).code(cache, code, indent);    \
        std::string rightCache = LoadCuda({right}).code(cache, code, indent);  \
        std::string outputCache =                                              \
            AllocateCuda({output}).code(cache, code, indent);                  \
        code += INDENTATION(indent) + outputCache + " = ";                     \
        if (CAST) {                                                            \
            code += "static_cast<" + dataTypeStr(dataType) + ">(";             \
        } else {                                                               \
            code += leftCache + " " + OP_STR + " " + rightCache;               \
        }                                                                      \
        if (CAST) {                                                            \
            code += ")";                                                       \
        }                                                                      \
        code += ";\n";                                                         \
        return "";                                                             \
    }

CUDA_BINARY_GENERATOR(Add, "+", false)
CUDA_BINARY_GENERATOR(Sub, "-", false)
CUDA_BINARY_GENERATOR(Mul, "*", false)
CUDA_BINARY_GENERATOR(Div, "/", false)
CUDA_BINARY_GENERATOR(Eq, "==", true)
CUDA_BINARY_GENERATOR(Ge, ">=", true)
CUDA_BINARY_GENERATOR(Gt, ">", true)
CUDA_BINARY_GENERATOR(Le, "<=", true)
CUDA_BINARY_GENERATOR(Lt, "<", true)
CUDA_BINARY_GENERATOR(Ne, "!=", true)
CUDA_BINARY_GENERATOR(And, "&", true)
CUDA_BINARY_GENERATOR(Or, "|", true)
CUDA_BINARY_GENERATOR(Xor, "^", true)
// CUDA_BINARY_GENERATOR(FloorMod, "floormod", false)
// CUDA_BINARY_GENERATOR(FloorDiv, "floordiv", false)

/**
 * Register Micros
 */
// CUDA
REGISTER_MICRO(OperatorType::ADD, Platform::CUDA, AddCuda::makeObj)
REGISTER_MICRO(OperatorType::SUB, Platform::CUDA, SubCuda::makeObj)
REGISTER_MICRO(OperatorType::MUL, Platform::CUDA, MulCuda::makeObj)
REGISTER_MICRO(OperatorType::DIV, Platform::CUDA, DivCuda::makeObj)
REGISTER_MICRO(OperatorType::EQ, Platform::CUDA, EqCuda::makeObj)
REGISTER_MICRO(OperatorType::GE, Platform::CUDA, GeCuda::makeObj)
REGISTER_MICRO(OperatorType::GT, Platform::CUDA, GtCuda::makeObj)
REGISTER_MICRO(OperatorType::LE, Platform::CUDA, LeCuda::makeObj)
REGISTER_MICRO(OperatorType::LT, Platform::CUDA, LtCuda::makeObj)
REGISTER_MICRO(OperatorType::NE, Platform::CUDA, NeCuda::makeObj)
REGISTER_MICRO(OperatorType::AND, Platform::CUDA, AndCuda::makeObj)
REGISTER_MICRO(OperatorType::OR, Platform::CUDA, OrCuda::makeObj)
REGISTER_MICRO(OperatorType::XOR, Platform::CUDA, XorCuda::makeObj)
// REGISTER_MICRO(OperatorType::FLOORMOD, Platform::CUDA, FloorModCuda::makeObj)
// REGISTER_MICRO(OperatorType::FLOORDIV, Platform::CUDA, FloorDivCuda::makeObj)

#undef CUDA_BINARY_GENERATOR

} // namespace infini
