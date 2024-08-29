#ifndef __BROADCAST_H__
#define __BROADCAST_H__
#include "core/common.h"
#include "core/operator.h"

namespace infini {

class Broadcast : public Operator {
  public:
    Broadcast(const std::vector<Tensor *> &inputs = {},
              const std::vector<Tensor *> &outputs = {},
              const Shape outShape = {}, const std::string &name = "");
    ~Broadcast() = default;
};

using BROADCAST = Broadcast;

} // namespace infini

#endif
