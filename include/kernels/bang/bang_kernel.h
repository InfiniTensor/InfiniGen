#pragma once
#include "core/api.h"
#include "core/kernel.h"

namespace infini{

class BangKernel: public Kernel {
  public:
  std::string generateCode(std::vector<std::string>& args) {
    return generateCodeOnBang(args);
  }
  virtual std::string generateCodeOnBang(std::vector<std::string>& args) const = 0;
};

}
