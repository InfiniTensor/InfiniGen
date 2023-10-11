#include "core/cache.h"
#include "core/utils.h"
#include "micros/binary_micro.h"
#include "micros/memory_micro.h"

namespace infini {

#define BANG_BIN_GENERATOR(OP, OP_STR)                                      \
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

BANG_BIN_GENERATOR(Add, "add")
BANG_BIN_GENERATOR(Sub, "sub")
BANG_BIN_GENERATOR(Mul, "mul")

/**
 * Register Micros
 */
// BANG
REGISTER_MICRO(OperatorType::ADD, Platform::BANG, AddBang::makeObj)
REGISTER_MICRO(OperatorType::SUB, Platform::BANG, SubBang::makeObj)
REGISTER_MICRO(OperatorType::MUL, Platform::BANG, MulBang::makeObj)

#undef BANG_BIN_GENERATOR

}  // namespace infini
