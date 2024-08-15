#ifndef OPERATOR_H
#define OPERATOR_H
#include "core/tensor.h"
#include "core/tile.h"
#include <unordered_map>
#include <utility>
#include <vector>

namespace infini {

class Tensor;

class Operator {
  private:
    // Number of ops created
    static int64_t operatorCount;

  public:
    /** Operator Information **/
    // Name of op
    std::string operatorName;
    // Index of op
    const int64_t operatorIndex;
    // Inputs of op
    std::vector<Tensor *> operatorInputs;
    // Temp variables of op
    std::vector<std::vector<Tensor *>> operatorTemps;
    // Outputs of op
    std::vector<Tensor *> operatorOutputs;

    /** Graph Information **/
    // Predecessor of this op in graph
    std::vector<Operator *> operatorPredecessors;
    // Successors of this op in graph
    std::vector<Operator *> operatorSuccessors;
    // Number of predecessor ops in graph
    int64_t operatorIndegree;
    // Number of outputs of this op (for future use)
    int64_t operatorOutputsNum;

  public:
    Operator() = delete;
    Operator(const std::vector<Tensor *> &inputs = {},
             const std::vector<Tensor *> &outputs = {},
             const std::string &name = "", const int64_t &outputsNum = 1);
    ~Operator() = default;

    std::string info(bool print = true);
    Tensor *getOutput(int64_t index);
    Tensor *getInput(int64_t index);
    std::vector<Tensor *> getOutputs();
    std::vector<Tensor *> getInputs();

    Operator *getPredecessor(int64_t index);
    Operator *getSuccessor(int64_t index);
    std::vector<Operator *> getPredecessors();
    std::vector<Operator *> getSuccessors();
};

} // namespace infini

#endif
