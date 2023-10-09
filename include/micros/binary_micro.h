#pragma once
#include "core/micro.h"

namespace infini {

// #ifndef MAKEOBJ
#define MAKEOBJ(MICRO)                                              \
  static Micro* makeObj(const std::vector<OperandType>& operands) { \
    ASSERT(operands.size() == 3);                                   \
    return new MICRO(operands);                                     \
  }
// #endif

#define BINARY_DEF(OP, PLName, PL)                                            \
  class CAT(OP, PLName) : public BinaryMicro {                                \
   public:                                                                    \
    CAT(OP, PLName)                                                           \
    (const std::vector<OperandType>& operands) : BinaryMicro(operands, PL) {} \
    std::string generatorCode(Cache& cache, std::string& code,                \
                              int64_t indent);                                \
    MAKEOBJ(CAT(OP, PLName))                                                  \
  }

class BinaryMicro : public Micro {
 protected:
  std::string left_name, right_name, output_name;
  int64_t left_offset, right_offset, output_offset;
  int64_t length;
  TensorDatatype data_type;

 public:
  BinaryMicro(const std::vector<OperandType>& operands, Platform pt)
      : Micro(MicroType::BINARY, pt),
        output_name(std::get<0>(operands[0])),
        left_name(std::get<0>(operands[1])),
        right_name(std::get<0>(operands[2])),
        output_offset(std::get<1>(operands[0])),
        left_offset(std::get<1>(operands[1])),
        right_offset(std::get<1>(operands[2])),
        length(std::get<2>(operands[0])),
        data_type(std::get<3>(operands[0])) {}
};

/**
 * Cuda Micro declearation
 *  1. AddCuda
 *  2. SubCuda
 *  3. MulCuda
 */
BINARY_DEF(Add, Cuda, Platform::CUDA);
BINARY_DEF(Sub, Cuda, Platform::CUDA);
BINARY_DEF(Mul, Cuda, Platform::CUDA);

/**
 * Bang Micro declaration
 *  1. AddBang
 *  2. SubBang
 *  3. MulBang
 */
BINARY_DEF(Add, Bang, Platform::BANG);
BINARY_DEF(Sub, Bang, Platform::BANG);
BINARY_DEF(Mul, Bang, Platform::BANG);

#undef MAKEOBJ
#undef BINARY_DEF
}  // namespace infini
