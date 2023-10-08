#pragma once
#include "core/micro.h"

namespace infini {

// #ifndef MAKEOBJ
#define MAKEOBJ(MICRO)                                              \
  static Micro* makeObj(const std::vector<OperandType>& operands) { \
    ASSERT(operands.size() == 1);                                   \
    return new MICRO(operands[0]);                                  \
  }
// #endif

#define MEMORY_DEF(OP, PLName, PL)                                            \
  class CAT(OP, PLName) : public MemoryMicro {                                \
   public:                                                                    \
    CAT(OP, PLName)(const OperandType& operand) : MemoryMicro(operand, PL) {} \
    std::string generatorCode(Cache& cache, std::string& code,                \
                              int64_t indent) override;                       \
    MAKEOBJ(CAT(OP, PLName))                                                  \
  }

class MemoryMicro : public Micro {
 protected:
  int64_t offset, length;
  std::string name;
  TensorDatatype data_type;

 public:
  MemoryMicro(const OperandType& operand, Platform pt)
      : Micro(MicroType::MEMORY, pt),
        name(std::get<0>(operand)),
        offset(std::get<1>(operand)),
        length(std::get<2>(operand)),
        data_type(std::get<3>(operand)) {}
  virtual std::string generatorCode(Cache& cache, std::string& code,
                                    int64_t indent = 0) = 0;
  static Micro* makeObj() { return nullptr; }
};

/**
 * Cuda memory micro implementation, including
 *  1. LoadCuda
 *  2. AllocateCuda
 *  3. StoreCuda
 *  4. FreeCuda
 */
MEMORY_DEF(Load, Cuda, Platform::CUDA);
MEMORY_DEF(Allocate, Cuda, Platform::CUDA);
MEMORY_DEF(Store, Cuda, Platform::CUDA);
MEMORY_DEF(Free, Cuda, Platform::CUDA);

/**
 * Bang memory micro implementation, including
 *  1. LoadBang
 *  2. AllocateBang
 *  3. StoreBang
 *  4. FreeBang
 */
MEMORY_DEF(Load, Bang, Platform::BANG);
MEMORY_DEF(Allocate, Bang, Platform::BANG);
MEMORY_DEF(Store, Bang, Platform::BANG);
MEMORY_DEF(Free, Bang, Platform::BANG);

/**
 * Register Micros withc operator type and platform
 */
// CUDA
REGISTER_MICRO(OperatorType::LOAD, Platform::CUDA, LoadCuda::makeObj)
REGISTER_MICRO(OperatorType::ALLOCATE, Platform::CUDA, AllocateCuda::makeObj)
REGISTER_MICRO(OperatorType::STORE, Platform::CUDA, StoreCuda::makeObj)
REGISTER_MICRO(OperatorType::FREE, Platform::CUDA, FreeCuda::makeObj)
// BANG
REGISTER_MICRO(OperatorType::LOAD, Platform::BANG, LoadBang::makeObj)
REGISTER_MICRO(OperatorType::ALLOCATE, Platform::BANG, AllocateBang::makeObj)
REGISTER_MICRO(OperatorType::STORE, Platform::BANG, StoreBang::makeObj)
REGISTER_MICRO(OperatorType::FREE, Platform::BANG, FreeBang::makeObj)

#undef MAKEOBJ
#undef MEMORY_DEF

}  // namespace infini
