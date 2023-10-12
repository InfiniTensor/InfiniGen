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
BANG_BIN_GENERATOR(Eq, "eq")
BANG_BIN_GENERATOR(Ge, "ge")
BANG_BIN_GENERATOR(Gt, "gt")
BANG_BIN_GENERATOR(Le, "le")
BANG_BIN_GENERATOR(Lt, "lt")
BANG_BIN_GENERATOR(Ne, "ne")
BANG_BIN_GENERATOR(And, "and")
BANG_BIN_GENERATOR(Or, "or")
BANG_BIN_GENERATOR(Xor, "xor")

// Div
std::string DivBang::generatorCode(Cache& cache, std::string& code,
                                   int64_t indent) {
  cache.lock();
  std::string left_cache =
      LoadBang(OperandType{left_name, left_offset, length, data_type})
          .generatorCode(cache, code, indent);
  std::string right_cache =
      LoadBang(OperandType{right_name, right_offset, length, data_type})
          .generatorCode(cache, code, indent);
  std::string output_cache =
      AllocateBang(OperandType{output_name, output_offset, length, data_type})
          .generatorCode(cache, code, indent);
  auto recip =
      OperandType{right_name + "_recip", right_offset, length, data_type};
  std::string recip_cache =
      AllocateBang(recip).generatorCode(cache, code, indent);
  code += indentation(indent) + "__bang_active_reciphp(" + recip_cache + ", " +
          right_cache + ", " + std::to_string(length) + ");\n";
  code += indentation(indent) + "__bang_mul(" + output_cache + ", " +
          left_cache + ", " + recip_cache + ", " + std::to_string(length) +
          ");\n";
  cache.unlock();
  FreeBang(recip).generatorCode(cache, code, indent);
  return "";
}

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

#undef BANG_BIN_GENERATOR

}  // namespace infini
