#pragma once
#include "core/type.h"
#include "core/tile.h"
#include "core/utils.h"
#include <vector>
#include <map>

namespace infini {

using KernelAttrs = std::tuple<PlatformType, KernelType>;  // platfrom , type

class Kernel {
  /**
   * @brief Kernel is smallest unit of codegen.
   *  It includes different types of memory access,
   *  operation codegen, and others
   */
 protected:
  KernelType kernel_type;
  PlatformType platform;

 public:
  // Constructor
  Kernel(){};
  Kernel(const Kernel &) = delete;
  // Destructor
  virtual ~Kernel(){};

  /**
   * @brief Generate code. Subclasses rewrite this
   * function and call generations which belongs to
   * specific platform.
   */
  virtual std::string generateCode(std::vector<std::string> &args) const = 0;

  /** @brief Information print*/
  virtual void printInformation();
};

class KernelRegistry {
  /**
   * @brief KernelRistry is a single Instancing class
   * which contains different kinds of kernels
   */
 public:
  using KernelRecord = std::tuple<Kernel *const, const std::string,
                                  const int>;  // Kernel* , name, ID

 private:
  std::map<KernelAttrs, KernelRecord> kernels;
  int nKernels = 0;

 public:
  // Deconstruct
  ~KernelRegistry() {
    for (auto &[k, v] : kernels) {
      delete std::get<0>(v);
    }
  }
  // Static function that makes KernelRegistry a class with single instance
  static KernelRegistry &getInstance() {
    static KernelRegistry instance;
    return instance;
  }
  // Register Kernel
  bool RegisterKernel(const KernelAttrs &key, Kernel *kernel,
                      std::string name) {
    ASSERT(kernels.find(key) == kernels.end());  // kernel not registered
    kernels.emplace(key, KernelRecord{kernel, name, ++nKernels});
    return true;
  }
  // Get kernel by kernelAttrs {PlatformType, kernelType}
  Kernel *getKernel(const KernelAttrs &kernelAttrs) const {
    auto it = kernels.find(kernelAttrs);
    ASSERT(it != kernels.end());
    return std::get<0>(it->second);
  }

  const KernelRecord &getKernelRecord(const KernelAttrs &kernelAttrs) {
    return kernels.at(kernelAttrs);
  }
};
}  // namespace infini

#define REGISTER_KERNEL(platform, kernelType, kernel, name) \
  namespace infini {                                        \
  static const bool _register_kernel_##__COUNTER__ =        \
      KernelRegistry::getInstance().registerKernel(         \
          KernelAttrs{platform, kernelType}, kernel, name); \
  }