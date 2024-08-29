#include "core/operator.h"
#include "core/utils.h"

namespace infini {

int64_t Operator::operatorCount = 0;

Operator::Operator(const OperatorType &type,
                   const std::vector<Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs,
                   const std::string &name, const int64_t &outputsNum)
    : operatorType(type), operatorInputs(inputs), operatorOutputs(outputs),
      operatorName(name == "" ? "Operator_" + std::to_string(operatorCount)
                              : name),
      operatorIndex(operatorCount++), operatorIndegree(0),
      operatorOutputsNum(outputsNum) {
    // Infer output shape and datatype in derived class
    // if (outputs.empty()) {
    //     Tensor *temp;
    //     for (auto i = 0; i < outputsNum; ++i) {
    //         temp =
    //             new Tensor(inputs[0]->tensorShape,
    //             inputs[0]->tensorDataType);
    //         operatorOutputs.push_back(temp);
    //     }
    // }
    for (auto it : operatorInputs) {
        it->addConsumer(this);
        it->tensorUsesLeft += 1;
        if (it->tensorProducer != nullptr) {
            operatorPredecessors.push_back(it->tensorProducer);
            it->tensorProducer->operatorSuccessors.push_back(this);
        }
    }
    for (auto it : operatorOutputs) {
        it->setProducer(this);
    }
    for (auto it : operatorInputs) {
        operatorIndegree += it->tensorProducer == NULL ? 0 : 1;
    }
}

std::string Operator::info(bool print) {
    std::stringstream out;
    out << BRIGHT_CYAN << HIGHLIGHT << "[OPERATOR] " << RESET;

    out << operatorName << ": [";
    for (auto i = 0; i < operatorInputs.size(); ++i) {
        out << operatorInputs[i]->tensorName;
        out << (i == (operatorInputs.size() - 1) ? "" : ", ");
    }
    out << "] --> (" << TO_STRING(operatorType) << ") --> [";
    for (auto i = 0; i < operatorOutputs.size(); ++i) {
        out << operatorOutputs[i]->tensorName;
        out << (i == (operatorOutputs.size() - 1) ? "" : ", ");
    }
    out << "], Pred: [";
    for (auto i = 0; i < operatorPredecessors.size(); ++i) {
        out << operatorPredecessors[i]->operatorName;
        out << (i == (operatorPredecessors.size() - 1) ? "" : ", ");
    }
    out << "], Succ: [";
    for (auto i = 0; i < operatorSuccessors.size(); ++i) {
        out << operatorSuccessors[i]->operatorName;
        out << (i == (operatorSuccessors.size() - 1) ? "" : ", ");
    }
    out << "]";

    if (print) {
        LOG(INFO) << out.str();
    }
    return out.str();
}

Tensor *Operator::getOutput(int64_t index) { return operatorOutputs[index]; }

Tensor *Operator::getInput(int64_t index) { return operatorInputs[index]; }

std::vector<Tensor *> Operator::getOutputs() { return operatorOutputs; }

std::vector<Tensor *> Operator::getInputs() { return operatorInputs; }

Operator *Operator::getPredecessor(int64_t index) {
    return operatorPredecessors[index];
}

Operator *Operator::getSuccessor(int64_t index) {
    return operatorSuccessors[index];
}

std::vector<Operator *> Operator::getPredecessors() {
    return operatorPredecessors;
}

std::vector<Operator *> Operator::getSuccessors() { return operatorSuccessors; }

void Operator::setAttribute(std::string key, Attribute attribute) {
    operatorAttributes[key] = attribute;
}

Attribute Operator::getAttribute(std::string key) {
    auto iter = operatorAttributes.find(key);
    CHECK(iter != operatorAttributes.end(), "Can't find this key: " + key);
    return iter->second;
}

void Operator::deleteAttribute(std::string key) {
    auto iter = operatorAttributes.find(key);
    if (iter != operatorAttributes.end()) {
        operatorAttributes.erase(iter);
    }
}
} // namespace infini
