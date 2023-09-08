#include "graphs/binary_unary_graph.h"
#include "core/utils.h"

namespace infini {

BinaryUnaryGraph::BinaryUnaryGraph(std::vector<Node*> operators_list,
                                   std::vector<Data*> inputs_list,
                                   std::vector<Data*> outputs_list,
                                   std::string name_value)
    : Graph(operators_list, inputs_list, outputs_list, name_value) {}

void BinaryUnaryGraph::generatorCode() {
  // TODO(wanghailu)
  LOG(INFO) << "1111";
}

}  // namespace infini
