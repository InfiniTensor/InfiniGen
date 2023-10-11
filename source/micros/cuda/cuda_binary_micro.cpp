#include "core/utils.h"
#include "core/cache.h"
#include "micros/memory_micro.h"
#include "micros/binary_micro.h"

namespace infini {

#define CUDA_GENERATOR(OP, OP_STR)                                            \
  std::string CAT(OP, Cuda)::generatorCode(Cache& cache, std::string& code,   \
                                           int64_t indent) {                  \
    cache.lock();                                                             \
    std::string left_cache =                                                  \
        LoadCuda(OperandType{left_name, left_offset, length, data_type})      \
            .generatorCode(cache, code, indent);                              \
    std::string right_cache =                                                 \
        LoadCuda(OperandType{right_name, right_offset, length, data_type})    \
            .generatorCode(cache, code, indent);                              \
    std::string output_cache =                                                \
        AllocateCuda(                                                         \
            OperandType{output_name, output_offset, length, data_type})       \
            .generatorCode(cache, code, indent);                              \
    code += indentation(indent) + output_cache + "] = " + left_cache + "] " + \
            std::string(OP_STR) + " " + right_cache + "];\n";                 \
    cache.unlock();                                                           \
    return "";                                                                \
  }

CUDA_GENERATOR(Add, "+")
CUDA_GENERATOR(Sub, "-")
CUDA_GENERATOR(Mul, "*")
CUDA_GENERATOR(Div, "/")
CUDA_GENERATOR(Eq, "=")
CUDA_GENERATOR(Ge, ">=")
CUDA_GENERATOR(Gt, ">")
CUDA_GENERATOR(Le, "<=")
CUDA_GENERATOR(Lt, "<")
CUDA_GENERATOR(Ne, "!=")
CUDA_GENERATOR(And, "&")
CUDA_GENERATOR(Or, "|")
CUDA_GENERATOR(Xor, "^")

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
