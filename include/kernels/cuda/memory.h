#pragma once
#include "core/kernel.h"

namespace infini {

#ifndef CUDA_KERNEL
#define CUDA_KERNEL(prefix, kernel_type)                     \
  class prefix##CudaKernel : public MemCudaKernel {       \
   public:                                                   \
    prefix##CudeKernel(KernelType kt = kernel_type,          \
                       PlatformType pt = PlatformType::CUDA) \
        : BinaryCudaKernel(kt, pt) {}                        \
    std::string generateCodeOnCuda(                          \
        std::vector<std::string>& args) const override;      \
  };

#undef CUDA_KERNEL
#endif

}  // namespace infini
