#pragma once
#include "core/graph.h"

namespace infini {

class Gemm : public Node {
 public:
  // Constructor
  Gemm(OperatorType type = OperatorType::GEMM,
       std::vector<Data *> inputs_list = {},
       std::vector<Data *> outputs_list = {}, std::string name_value = "",
       int64_t outputs_num_value = 1);
  // Destructor
  ~Gemm() = default;
};

}  // namespace infini