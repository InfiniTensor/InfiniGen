#ifndef __UNARY_H__
#define __UNARY_H__
#include "core/operator.h"

namespace infini {
class Unary : public Operator {
  public:
    Unary(const OperatorType &type, const std::vector<Tensor *> &inputs = {},
          const std::vector<Tensor *> &outputs = {},
          const std::string &name = "");
    ~Unary() = default;
};

#define DEFINE_UNARY(OP_NAME)                                                  \
    class OP_NAME : public Unary {                                             \
      public:                                                                  \
        OP_NAME(const std::vector<Tensor *> &inputs = {},                      \
                const std::vector<Tensor *> &outputs = {},                     \
                const std::string &name = "")                                  \
            : Unary(OperatorType::OP_NAME, inputs, outputs, name) {}           \
    };

DEFINE_UNARY(SQRT)
DEFINE_UNARY(RSQRT)
DEFINE_UNARY(RELU)
DEFINE_UNARY(RECIP)
DEFINE_UNARY(SIGMOID)
DEFINE_UNARY(SIN)
DEFINE_UNARY(COS)
DEFINE_UNARY(TANH)
#undef DEFINE_UNARY

} // namespace infini

#endif
