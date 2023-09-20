#pragma once
#include "core/micro.h"

namespace infini {

#define BINARY_MICRO(MICRO_NAME, MICRO_TYPE, PLATFORM_TYPE)                  \
  class MICRO_NAME##Micro : public Micro {                                   \
    int64_t output, left, right, length;                                     \
    std::string output_name, left_name, right_name;                          \
                                                                             \
   public:                                                                   \
    MICRO_NAME##Micro(std::string output_name_string, int64_t output_offset, \
                      std::string left_name_string, int64_t left_offset,     \
                      std::string right_name_string, int64_t right_offset,   \
                      int64_t length_value)                                  \
        : output_name(output_name_string),                                   \
          output(output_offset),                                             \
          left_name(left_name_string),                                       \
          left(left_offset),                                                 \
          right_name(right_name_string),                                     \
          right(right_offset),                                               \
          Micro(MICRO_TYPE, PLATFORM_TYPE),                                  \
          length(length_value) {}                                            \
    std::string generatorCode(Cache& cache, std::string& result,             \
                              std::string coreIndex = "") override;          \
  };

// On BANG platform
BINARY_MICRO(BangAdd, MicroType::ADD, PlatformType::BANG)
BINARY_MICRO(BangSub, MicroType::SUB, PlatformType::BANG)
BINARY_MICRO(BangMul, MicroType::MUL, PlatformType::BANG)

// On Cuda platform
BINARY_MICRO(CudaAdd, MicroType::ADD, PlatformType::CUDA)
BINARY_MICRO(CudaSub, MicroType::SUB, PlatformType::CUDA)
BINARY_MICRO(CudaMul, MicroType::MUL, PlatformType::CUDA)

#undef BINARY_MICRO
}  // namespace infini
