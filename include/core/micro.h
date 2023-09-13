#pragma once
#include <vector>
#include <string>
#include "core/cache.h"

namespace infini {

class Micro {
 public:
  // Constructor
  Micro() = default;
  // Destructor
  ~Micro() = default;
  // Generator
  virtual std::string generatorCode(Cache& cache, std::string& result);
};

}  // namespace infini
