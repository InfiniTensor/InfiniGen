#pragma once
#include "core/type.h"
#include "core/cache.h"
#include <string>

namespace infini {

// TODO: delete line after change to Kernel
using MicroType = KernelType;

class Micro {
  /**
   * @brief Micro is smallest unit of codegen.
   *  It includes different types of memory access,
   *  operation codegen, and others
   */
 protected:
  MicroType micro_type;
  PlatformType platform;

 public:
  // Constructor
  Micro(){};
  Micro(const Micro &) = delete;
  Micro(MicroType mt, PlatformType pt);
  // Destructor
  virtual ~Micro(){};

  /**
   * @brief Generate code. Subclasses rewrite this
   * function and call generations which belongs to
   * specific platform.
   */
  virtual std::string generatorCode(Cache &cache, std::string &result,
                                    std::string coreIndex = "",
                                    int64_t indent = 0) = 0;

  /** @brief Information print*/
  virtual void printInformation();
};

}  // namespace infini
