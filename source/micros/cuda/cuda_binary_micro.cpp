#include "core/utils.h"
#include "core/cache.h"
#include "micros/memory_micro.h"
#include "micros/binary_micro.h"

namespace infini {

#define CUDA_GENERATOR(OP, OP_STR, CAST)                                       \
  std::string CAT(OP, Cuda)::generatorCode(Cache &cache, std::string &code,    \
                                           int64_t indent) {                   \
    cache.lock();                                                              \
    std::string left_cache =                                                   \
        LoadCuda(OperandType{left_name, left_offset, length, data_type})       \
            .generatorCode(cache, code, indent);                               \
    std::string right_cache =                                                  \
        LoadCuda(OperandType{right_name, right_offset, length, data_type})     \
            .generatorCode(cache, code, indent);                               \
    std::string output_cache =                                                 \
        AllocateCuda(                                                          \
            OperandType{output_name, output_offset, length, data_type})        \
            .generatorCode(cache, code, indent);                               \
    code += indentation(indent) + output_cache + "] = ";                       \
    if (CAST) {                                                                \
      code += "static_cast<" + datatype_string(data_type) + ">(";              \
    }                                                                          \
    code += left_cache + "] " + std::string(OP_STR) + " " + right_cache + "]"; \
    if (CAST) {                                                                \
      code += ")";                                                             \
    }                                                                          \
    code += ";\n";                                                             \
    cache.unlock();                                                            \
    return "";                                                                 \
  }

CUDA_GENERATOR(Add, "+", false)
CUDA_GENERATOR(Sub, "-", false)
CUDA_GENERATOR(Mul, "*", false)
CUDA_GENERATOR(Div, "/", false)
CUDA_GENERATOR(Eq, "==", true)
CUDA_GENERATOR(Ge, ">=", true)
CUDA_GENERATOR(Gt, ">", true)
CUDA_GENERATOR(Le, "<=", true)
CUDA_GENERATOR(Lt, "<", true)
CUDA_GENERATOR(Ne, "!=", true)
CUDA_GENERATOR(And, "&", true)
CUDA_GENERATOR(Or, "|", true)
CUDA_GENERATOR(Xor, "^", true)

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

#undef CUDA_GENERATOR

}  // namespace infini
