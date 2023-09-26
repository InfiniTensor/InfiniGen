#pragma once
#include "core/graph.h"

namespace infini {

class BinaryUnaryGraph : public Graph {
 public:
  // Constructor
  BinaryUnaryGraph(std::vector<Node*> operators_list = {},
                   std::vector<Data*> inputs_list = {},
                   std::vector<Data*> outputs_list = {},
                   std::string name_value = "");
  // Destructor
  ~BinaryUnaryGraph() = default;
  // Generator
  std::string generatorTask(int64_t indent) override;
  std::string generatorHost(int64_t indent) override;
  std::string generatorCode(int64_t indent) override;
  void applyPlatform(Platform platform) override;
};

}  // namespace infini
