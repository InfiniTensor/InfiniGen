#pragma once
#include "core/micro.h"

namespace infini {

#define BINARY_MICRO(MICRO_NAME, MICRO_TYPE, PLATFORM_TYPE)                  \
  class MICRO_NAME##Micro : public Micro {                                   \
    int64_t output, left, right, length;                                     \
    std::string output_name, left_name, right_name;                          \
    TensorDatatype data_type;                                                \
                                                                             \
   public:                                                                   \
    MICRO_NAME##Micro(std::string output_name_string, int64_t output_offset, \
                      std::string left_name_string, int64_t left_offset,     \
                      std::string right_name_string, int64_t right_offset,   \
                      int64_t length_value, TensorDatatype dtype)            \
        : Micro(MICRO_TYPE, PLATFORM_TYPE),                                  \
          output_name(output_name_string),                                   \
          output(output_offset),                                             \
          left_name(left_name_string),                                       \
          left(left_offset),                                                 \
          right_name(right_name_string),                                     \
          right(right_offset),                                               \
          length(length_value),                                              \
          data_type(dtype) {}                                                \
    std::string generatorCode(Cache& cache, std::string& result,             \
                              int64_t indent = 0) override;                  \
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
