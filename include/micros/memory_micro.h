#pragma once
#include "core/micro.h"

namespace infini {

#define MEMORY_MICRO(PLATFORM_NAME, OPNAME, MICRO_TYPE, PLATFORM_TYPE) \
  class OPNAME##PLATFORM_NAME : public Micro {                         \
    int64_t offset, length;                                            \
    std::string name;                                                  \
                                                                       \
   public:                                                             \
    OPNAME##PLATFORM_NAME(const std::vector<OperandType> &operands)    \
        : Micro(MICRO_TYPE, PLATFORM_TYPE),                            \
          name(std::get<0>(operands[0])),                              \
          offset(std::get<1>(operands[0])),                            \
          length(std::get<2>(operands[0])) {}                          \
    OPNAME##PLATFORM_NAME(const OperandType &operand)                  \
        : Micro(MICRO_TYPE, PLATFORM_TYPE),                            \
          name(std::get<0>(operand)),                                  \
          offset(std::get<1>(operand)),                                \
          length(std::get<2>(operand)) {}                              \
    std::string generatorCode(Cache &cache, std::string &result,       \
                              int64_t indent = 0) override;            \
    static Micro *makeObj(const std::vector<OperandType> &operands) {  \
      return new OPNAME##PLATFORM_NAME(operands);                      \
    }                                                                  \
  };

}  // namespace infini
