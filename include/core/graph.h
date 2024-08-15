#ifndef GRAPH_H
#define GRAPH_H
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infini {

class Graph {
  private:
    static int64_t graphCount;

  public:
    // Graph Information
    std::string graphName;
    const int64_t graphIndex;
    std::vector<Operator *> graphOperators;
    std::vector<Tensor *> graphInputs;
    std::vector<Tensor *> graphOutputs;
    std::vector<Tensor *> graphTemps;
    std::unordered_set<Tensor *> graphRemainingTensors;

  public:
    Graph(std::vector<Operator *> operators = {},
          std::vector<Tensor *> inputs = {}, std::vector<Tensor *> outputs = {},
          std::string name = "");
    ~Graph() = default;

    std::vector<Operator *> topoSort();

    std::string info(bool print = true);
};

} // namespace infini

#endif
