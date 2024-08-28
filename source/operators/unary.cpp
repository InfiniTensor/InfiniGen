#include "operators/unary.h"

namespace infini {
Unary::Unary(const OperatorType &type, const std::vector<Tensor *> &inputs,
             const std::vector<Tensor *> &outputs, const std::string &name,
             const int64_t &outputsNum)
    : Operator(type, inputs, outputs, name, 1) {}
} // namespace infini
