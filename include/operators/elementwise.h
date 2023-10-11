#pragma once
#include "core/graph.h"

namespace infini {

class Binary : public Node {
 public:
  // Constructor
  Binary(OperatorType type, std::vector<Data*> inputs_list = {},
         std::vector<Data*> outputs_list = {}, std::string name_value = "",
         int64_t outputs_num_value = 1);
  // Destructor
  ~Binary() = default;
};

#define DEFINE_BINARY(OP_NAME)                                                 \
  class OP_NAME : public Binary {                                              \
   public:                                                                     \
    OP_NAME(std::vector<Data*> inputs_list = {},                               \
            std::vector<Data*> outputs_list = {}, std::string name_value = "", \
            int64_t outputs_num_value = 1)                                     \
        : Binary(OperatorType::OP_NAME, inputs_list, outputs_list, name_value, \
                 outputs_num_value) {}                                         \
  };

DEFINE_BINARY(ADD)
DEFINE_BINARY(SUB)
DEFINE_BINARY(MUL)
#undef DEFINE_BINARY_OBJ

}  // namespace infini
