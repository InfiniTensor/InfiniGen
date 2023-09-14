#pragma once
#include "core/micro.h"

namespace infini {

#define MEMORY_MICRO(MICRO_NAME)                                           \
  class MICRO_NAME##Micro : public Micro {                                 \
    int64_t data, length;                                                  \
    std::string data_name;                                                 \
                                                                           \
   public:                                                                 \
    MICRO_NAME##Micro(std::string data_name_string, int64_t data_offset,   \
                      int64_t length_value)                                \
        : data_name(data_name_string),                                     \
          data(data_offset),                                               \
          length(length_value) {}                                          \
    std::string generatorCode(Cache& cache, std::string& result) override; \
  };

MEMORY_MICRO(BangLoad)
MEMORY_MICRO(BangStore)
MEMORY_MICRO(BangAllocate)
MEMORY_MICRO(BangFree)

MEMORY_MICRO(CudaLoad)
MEMORY_MICRO(CudaStore)
MEMORY_MICRO(CudaAllocate)
MEMORY_MICRO(CudaFree)

#undef MEMORY_MICRO
}  // namespace infini
