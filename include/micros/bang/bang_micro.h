#pragma once
#include "core/api.h"
#include "core/micro.h"
#include "core/cache.h"

namespace infini {

// TODO: delete this line after change to Kernel
using KernelType = MicroType;

class BangMicro : public Micro {
  /**
   * @brief BangMicro is the subclass of Micro
   * which is the abstract micro gernating bang code
   */
 public:
  BangMicro() {}
  BangMicro(const BangMicro &) = delete;
  virtual ~BangMicro() {}
  /**
   * @brief generateCode calls generateCodeOnBang
   * which specifies generating bang code
   */
  virtual std::string generateCode(Cache &cache,
                                   std::string &result) const override {
    return generateCodeOnBang(cache, result);
  }
  virtual std::string generateCodeOnBang(Cache &cache,
                                         std::string &result) const = 0;
};

#define BANG_MICRO(prefix, MICRO_TYPE)                                  \
  class prefix##BangMicro : public BangMicro {                          \
   public:                                                              \
    prefix##BangMicro(MicroType kt = MICRO_TYPE,                        \
                      PlatformType pt = PlatformType::BANG) {           \
      micro_type = kt;                                                  \
      platform = pt;                                                    \
    }                                                                   \
    std::string generateCodeOnBang(Cache &cache,                        \
                                   std::string &result) const override; \
  };  // namespace infini

}  // namespace infini
