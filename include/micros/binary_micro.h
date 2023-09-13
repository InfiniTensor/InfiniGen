#pragma once
#include "core/micro.h"

namespace infini {

#define BINARY_MICRO(MICRO_NAME)                                             \
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
          length(length_value) {}                                            \
    std::string generatorCode(Cache& cache, std::string& result) override;   \
  };

BINARY_MICRO(BangAdd)
#undef BINARY_MICRO
}  // namespace infini