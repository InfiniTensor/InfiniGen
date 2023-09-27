#pragma once
#include "core/micro.h"

namespace infini {

#define MEMORY_MICRO(MICRO_NAME, MICRO_TYPE, PLATFORM_TYPE)              \
  class MICRO_NAME##Micro : public Micro {                               \
    int64_t data, length;                                                \
    std::string data_name;                                               \
    TensorDatatype data_type;                                            \
                                                                         \
   public:                                                               \
    MICRO_NAME##Micro(std::string data_name_string, int64_t data_offset, \
                      int64_t length_value, TensorDatatype dtype)        \
        : Micro(MICRO_TYPE, PLATFORM_TYPE),                              \
          data_name(data_name_string),                                   \
          data(data_offset),                                             \
          length(length_value),                                          \
          data_type(dtype) {}                                            \
    std::string generatorCode(Cache &cache, std::string &result,         \
                              int64_t indent = 0) override;              \
  };

MEMORY_MICRO(BangLoad, MicroType::LOAD, PlatformType::BANG)
MEMORY_MICRO(BangStore, MicroType::STORE, PlatformType::BANG)
MEMORY_MICRO(BangAllocate, MicroType::ALLOCATE, PlatformType::BANG)
MEMORY_MICRO(BangFree, MicroType::FREE, PlatformType::BANG)

MEMORY_MICRO(CudaLoad, MicroType::LOAD, PlatformType::CUDA)
MEMORY_MICRO(CudaStore, MicroType::STORE, PlatformType::CUDA)
MEMORY_MICRO(CudaAllocate, MicroType::ALLOCATE, PlatformType::CUDA)
MEMORY_MICRO(CudaFree, MicroType::FREE, PlatformType::CUDA)

#undef MEMORY_MICRO
}  // namespace infini
