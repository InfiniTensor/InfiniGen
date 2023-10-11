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
                              int64_t indent) override;                       \
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
  virtual std::string generatorCode(Cache& cache, std::string& code,
                                    int64_t indent = 0) = 0;
  static Micro* makeObj() { return nullptr; }
};

/**
 * Cuda Micro declearation
 *  1. AddCuda
 *  2. SubCuda
 *  3. MulCuda
 *  4. DivCuda
 *  5. EqCuda
 *  6. GeCuda
 *  7. GtCuda
 *  8. LeCuda
 *  9. LtCuda
 * 10. NeCuda
 * 11. AndCuda
 * 12. OrCuda
 * 13. XorCuda
 */
BINARY_DEF(Add, Cuda, Platform::CUDA);
BINARY_DEF(Sub, Cuda, Platform::CUDA);
BINARY_DEF(Mul, Cuda, Platform::CUDA);
// BINARY_DEF(Div, Cuda, Platform::CUDA);
BINARY_DEF(Eq, Cuda, Platform::CUDA);
BINARY_DEF(Ge, Cuda, Platform::CUDA);
BINARY_DEF(Gt, Cuda, Platform::CUDA);
BINARY_DEF(Le, Cuda, Platform::CUDA);
BINARY_DEF(Lt, Cuda, Platform::CUDA);
BINARY_DEF(Ne, Cuda, Platform::CUDA);
BINARY_DEF(And, Cuda, Platform::CUDA);
BINARY_DEF(Or, Cuda, Platform::CUDA);
BINARY_DEF(Xor, Cuda, Platform::CUDA);

/**
 * Bang Micro declaration
 *  1. AddBang
 *  2. SubBang
 *  3. MulBang
 *  4. DivBang (Not supported on arch 372)
 *  5. EqBang
 *  6. GeBang
 *  7. GtBang
 *  8. LeBang
 *  9. LtBang
 * 10. NeBang
 * 11. AndBang
 * 12. OrBang
 * 13. XorBang
 */
BINARY_DEF(Add, Bang, Platform::BANG);
BINARY_DEF(Sub, Bang, Platform::BANG);
BINARY_DEF(Mul, Bang, Platform::BANG);
// BINARY_DEF(Div, Bang, Platform::BANG);
BINARY_DEF(Eq, Bang, Platform::BANG);
BINARY_DEF(Ge, Bang, Platform::BANG);
BINARY_DEF(Gt, Bang, Platform::BANG);
BINARY_DEF(Le, Bang, Platform::BANG);
BINARY_DEF(Lt, Bang, Platform::BANG);
BINARY_DEF(Ne, Bang, Platform::BANG);
BINARY_DEF(And, Bang, Platform::BANG);
BINARY_DEF(Or, Bang, Platform::BANG);
BINARY_DEF(Xor, Bang, Platform::BANG);

#undef MAKEOBJ
#undef BINARY_DEF
}  // namespace infini
