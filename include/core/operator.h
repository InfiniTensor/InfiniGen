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
    static int64_t operatorCount;

  public:
    // Operator Information
    std::string operatorName;
    const int64_t operatorIndex;
    std::vector<Tensor *> operatorInputs;
    std::vector<std::vector<Tensor *>> operatorTemps;
    std::vector<Tensor *> operatorOutputs;

    // Graph Information
    std::vector<Operator *> operatorPredecessors;
    std::vector<Operator *> operatorSuccessors;
    int64_t operatorIndegree;
    int64_t operatorOutputsNum;

    // std::unordered_map<std::string, Attribute> operatorAttributes;
    // Task *operator_task;

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