#pragma
#include "core/micro.h"
#include "core/utils.h"

namespace infini {

#define BINARY_MICRO(PLATFORM_NAME, MICRO_NAME, MICRO_TYPE, PLATFORM_TYPE) \
  class MICRO_NAME##PLATFORM_NAME : public Micro {                         \
   private:                                                                \
    std::string left_name, right_name, output_name;                        \
    int64_t left_offset, right_offset, output_offset;                      \
    int64_t length;                                                        \
                                                                           \
   public:                                                                 \
    MICRO_NAME##PLATFORM_NAME(const std::vector<OperandType>& operands)    \
        : Micro(MICRO_TYPE, PLATFORM_TYPE),                                \
          output_name(std::get<0>(operands[0])),                           \
          left_name(std::get<0>(operands[1])),                             \
          right_name(std::get<0>(operands[2])),                            \
          output_offset(std::get<1>(operands[0])),                         \
          left_offset(std::get<1>(operands[1])),                           \
          right_offset(std::get<1>(operands[2])),                          \
          length(std::get<2>(operands[0])) {}                              \
    std::string generatorCode(Cache& cache, std::string& result,           \
                              int64_t indent = 0) override;                \
    static Micro* makeObj(const std::vector<OperandType>& operands) {      \
      return new MICRO_NAME##PLATFORM_NAME(operands);                      \
    }                                                                      \
  };

}  // namespace infini
