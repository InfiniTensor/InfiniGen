#pragma once
#include "core/api.h"
#include "core/kernel.h"

namespace infini {

class CudaKernel : public Kernel {
 /**
  * @brief CudaKernel is the subclass of Kernel
  * which is the abstract kernel gernating cuda code
 */
 public:
  CudaKernel(){}
  CudaKernel(const CudaKernel &) = delete;
  virtual ~CudaKernel(){}
  /**
   * @brief generateCode calls generateCodeOnCuda
   * which specifies generating cuda code
  */
  virtual std::string generateCode(
      std::vector<std::string>& args) const override {
    return generateCodeOnCuda(args);
  }
  virtual std::string generateCodeOnCuda(
      std::vector<std::string>& args) const = 0;
};
}  // namespace infini

#define CUDA_KERNEL(prefix, KERNEL_TYPE)                       \
  class prefix##CudaKernel : public CudaKernel {               \
   public:                                                     \
    prefix##CudeKernel(KernelType kt = KERNEL_TYPE,            \
                       PlatformType pt = PlatformType::CUDA) { \
      kernel_type = KERNEL_TYPE;                               \
      platform = pt;                                           \
    }                                                          \
    std::string generateCodeOnCuda(                            \
        std::vector<std::string>& args) const override;        \
  };