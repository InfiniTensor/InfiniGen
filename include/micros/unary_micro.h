#pragma once
#include "core/micro.h"

namespace infini {

#define MAKEOBJ(MICRO)                                              \
  static Micro* makeObj(const std::vector<OperandType>& operands) { \
    ASSERT(operands.size() == 2);                                   \
    return new MICRO(operands);                                     \
  }

#define UNARY_DEF(OP, PLName, PL)                                            \
  class CAT(OP, PLName) : public UnaryMicro {                                \
   public:                                                                   \
    CAT(OP, PLName)                                                         \
    (const std::vector<OperandType>& operands) : UnaryMicro(operands, PL) {} \
    std::string generatorCode(Cache& cache, std::string& code,               \
                              int64_t indent) override;                      \
    MAKEOBJ(CAT(OP, PLName))                                                 \
  }

class UnaryMicro : public Micro {
 protected:
  std::string input_name, output_name;
  int64_t input_offset, output_offset;
  int64_t length;
  TensorDatatype data_type;

 public:
  UnaryMicro(const std::vector<OperandType>& operands, Platform pt)
      : Micro(MicroType::UNARY, pt),
        output_name(std::get<0>(operands[0])),
        input_name(std::get<0>(operands[1])),
        output_offset(std::get<1>(operands[0])),
        input_offset(std::get<1>(operands[1])),
        length(std::get<2>(operands[0])),
        data_type(std::get<3>(operands[0])) {}
  virtual std::string generatorCode(Cache& cache, std::string& code,
                                    int64_t indent = 0) = 0;
  static Micro* makeObj() { return nullptr; }
};

/**
 * Cuda Unary micros
 *  1. SqrtCuda
 *  2. SigmoidCuda
 *  3. SoftmaxCuda
 *  4. ReluCuda
*/
UNARY_DEF(Sqrt, Cuda, Platform::CUDA);
UNARY_DEF(Sigmoid, Cuda, Platform::CUDA);
UNARY_DEF(Relu, Cuda, Platform::CUDA);
UNARY_DEF(RSqrt, Cuda, Platform::CUDA);
UNARY_DEF(Recip, Cuda, Platform::CUDA);

/**
 * Bang Unary micros
 *  1. SqrtBang
 *  2. SigmoidBang
 *  3. SoftmaxBang
 *  4. ReluBang
*/
UNARY_DEF(Sqrt, Bang, Platform::BANG);
UNARY_DEF(Sigmoid, Bang, Platform::BANG);
UNARY_DEF(Relu, Bang, Platform::BANG);
UNARY_DEF(RSqrt, Bang, Platform::BANG);
UNARY_DEF(Recip, Bang, Platform::BANG);

#undef MAKEOBJ
#undef UNARY_DEF

}  // namespace infini