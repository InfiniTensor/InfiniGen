#pragma once
#include "core/kernel.h"

namespace infini {

class BinaryKernel : public Kernel {
 public:
  // Constructor
  BinaryKernel();
  // Destructor
  ~BinaryKernel() = default;
};

#define BINARY_KERNEL(KERNEL_NAME)                                           \
  class KERNEL_NAME##Kernel : public BinaryKernel {                          \
   public:                                                                   \
    KERNEL_NAME##Kernel() : BinaryKernel() {}                                \
    std::string generatorCodeOnCUDA(std::vector<std::string> args) override; \
    std::string generatorCodeOnBANG(std::vector<std::string> args) override; \
  };

BINARY_KERNEL(ADD)
BINARY_KERNEL(SUB)
BINARY_KERNEL(MUL)
#undef BINARY_KERNEL
}  // namespace infini
