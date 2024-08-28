#ifndef __BINRAY_H__
#define __BINRAY_H__
#include "core/operator.h"

namespace infini {
class Binary : public Operator {
  public:
    Binary(const OperatorType &type, const std::vector<Tensor *> &inputs = {},
           const std::vector<Tensor *> &outputs = {},
           const std::string &name = "", const int64_t &outputsNum = 1);
    ~Binary() = default;
};

#define DEFINE_BINARY(OP_NAME)                                                 \
    class OP_NAME : public Binary {                                            \
      public:                                                                  \
        OP_NAME(const std::vector<Tensor *> &inputs = {},                      \
                const std::vector<Tensor *> &outputs = {},                     \
                const std::string &name = "", const int64_t &outputsNum = 1)   \
            : Binary(OperatorType::OP_NAME, inputs, outputs, name,             \
                     outputsNum) {}                                            \
    };

DEFINE_BINARY(ADD)
DEFINE_BINARY(SUB)
DEFINE_BINARY(MUL)
DEFINE_BINARY(DIV)
DEFINE_BINARY(EQ)
DEFINE_BINARY(GE)
DEFINE_BINARY(GT)
DEFINE_BINARY(LE)
DEFINE_BINARY(LT)
DEFINE_BINARY(NE)
DEFINE_BINARY(AND)
DEFINE_BINARY(OR)
DEFINE_BINARY(XOR)
DEFINE_BINARY(FLOORMOD)
DEFINE_BINARY(FLOORDIV)
#undef DEFINE_BINARY

} // namespace infini

#endif
