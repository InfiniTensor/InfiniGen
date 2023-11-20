#include "operators/gemm.h"

namespace infini {

Gemm::Gemm(OperatorType type, std::vector<Data *> inputs_list,
           std::vector<Data *> outputs_list, std::string name_value,
           int64_t outputs_num_value)
    : Node(inputs_list, outputs_list, name_value, outputs_num_value) {
  operator_type = type;
}

}  // namespace infini