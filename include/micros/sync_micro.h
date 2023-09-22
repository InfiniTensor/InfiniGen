#pragma once
#include "core/micro.h"

namespace infini {

#define SYNC_MICRO(MICRO_NAME, MICRO_TYPE, PLATFORM_TYPE)           \
  class MICRO_NAME##Micro : public Micro {                          \
   public:                                                          \
    MICRO_NAME##Micro() {}                                          \
    std::string generatorCode(Cache& cache, std::string& result,    \
                              std::string coreIndex = "") override; \
  };

SYNC_MICRO(BangSync, MicroType::SYNC, PlatformType::BANG)
SYNC_MICRO(CudaSync, MicroType::SYNC, PlatformType::CUDA)

#undef SYNC_MICRO
}  // namespace infini
