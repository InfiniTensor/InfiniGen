#pragma once
#include "core/kernel.h"

namespace infini {

class MemoryKernel : public Kernel {
 public:
  // Constructor
  MemoryKernel();
  // Destructor
  ~MemoryKernel() = default;
};

#define MEMORY_KERNEL(KERNEL_NAME)                                           \
  class KERNEL_NAME##Kernel : public MemoryKernel {                          \
   public:                                                                   \
    KERNEL_NAME##Kernel() : MemoryKernel() {}                                \
    std::string generatorCodeOnCUDA(std::vector<std::string> args) override; \
    std::string generatorCodeOnBANG(std::vector<std::string> args) override; \
  };

MEMORY_KERNEL(G2R)
MEMORY_KERNEL(R2G)
#undef MEMORY_KERNEL
}  // namespace infini
