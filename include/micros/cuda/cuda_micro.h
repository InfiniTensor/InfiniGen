#pragma
#include "core/cache.h"
#include "core/micro.h"
#include "core/api.h"

namespace infini {

// TODO: delete this line after change to Kernel
using MicroType = KernelType;

class CudaMicro : public Micro {
  /**
   * @brief CudaMicro is subclass of Micro
   * which is the abstract of generating cuda code
   */
 public:
  CudaMicro() {}
  CudaMicro(const CudaMicro&) = delete;
  virtual ~CudaMicro() {}

  virtual std::string generatorCode(Cache& cache,
                                    std::string& result) const override {
    return generateCodeOnCuda(cache, result);
  }
  virtual std::string generateCodeOnCuda(Cache& cache,
                                         std::string& result) const = 0;
};

#define CUDA_MICRO(prefix, MICRO_TYPE)                                  \
  class prefix##CudaMicro : public CudaMicro {                          \
   public:                                                              \
    prefix##CudaMicro(MicroType mt, PlatformType pt) {                  \
      micro_type = mt;                                                  \
      platform = pt;                                                    \
    }                                                                   \
    std::string generateCodeOnCuda(Cache& cache,                        \
                                   std::string& result) const override; \
  };

}  // namespace infini
