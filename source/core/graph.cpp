#include "core/graph.h"
#include "core/utils.h"
#include <algorithm>

namespace infini {

int64_t Graph::graphCount = 0;

Graph::Graph(std::vector<Operator *> operators, std::vector<Tensor *> inputs,
             std::vector<Tensor *> outputs, std::string name)
    : graphOperators(operators), graphInputs(inputs), graphOutputs(outputs),
      graphName(name == "" ? "Graph_" + std::to_string(graphCount) : name),
      graphIndex(graphCount++) {
    for (auto op : graphOperators) {
        for (auto tensor : op->operatorInputs) {
            graphRemainingTensors.insert(tensor);
        }
        for (auto tensor : op->operatorOutputs) {
            auto it =
                std::find(graphOutputs.begin(), graphOutputs.end(), tensor);
            if (it == graphOutputs.end()) {
                graphTemps.push_back(tensor);
            }
        }
    }
}

std::string Graph::info(bool print) {
    std::stringstream out;
    out << BRIGHT_MAGENTA << HIGHLIGHT << "[GRAPH] " << RESET;
    std::string prefix = out.str();
    out << graphName << std::endl;

    auto opSorted = topoSort();
    auto getTensorName =
        [](std::vector<Tensor *> tensors) -> std::vector<std::string> {
        std::vector<std::string> res;
        for (auto tensor : tensors) {
            res.push_back(tensor->tensorName);
        }
        return res;
    };
    std::vector<std::string> graphInputNames = getTensorName(graphInputs);
    std::vector<std::string> graphOutputNames = getTensorName(graphOutputs);
    for (auto op : opSorted) {
        auto opInputs = op->getInputs();
        auto opOutputs = op->getOutputs();
        std::vector<std::string> opInputNames = getTensorName(opInputs);
        std::vector<std::string> opOutputNames = getTensorName(opOutputs);
        auto printTensors = [](std::vector<std::string> tensors,
                               std::vector<std::string> list) -> std::string {
            std::string res = "[";
            for (auto i = 0; i < tensors.size(); ++i) {
                if (std::find(list.begin(), list.end(), tensors[i]) !=
                    list.end()) {
                    res += GREEN + tensors[i] + RESET;
                } else {
                    res += tensors[i];
                }
                res += (i == (tensors.size() - 1) ? "" : ", ");
            }
            res += "]";
            return res;
        };
        out << prefix << "\t" << printTensors(opInputNames, graphInputNames)
            << " --> (" << op->operatorName << ") --> "
            << printTensors(opOutputNames, graphOutputNames) << "\n";
    }

    if (print) {
        std::istringstream iss(out.str());
        std::string line;
        while (std::getline(iss, line)) {
            LOG(INFO) << line;
        }
    }
    return out.str();
}

std::vector<Operator *> Graph::topoSort() {
    std::unordered_map<Operator *, int64_t> operatorList;
    for (auto op : graphOperators) {
        operatorList[op] = op->operatorIndegree;
    }
    std::vector<Operator *> result;
    while (!operatorList.empty()) {
        for (auto op = operatorList.begin(); op != operatorList.end(); ++op) {
            if (op->second == 0) {
                result.push_back(op->first);
                for (auto successor : (op->first)->operatorSuccessors) {
                    --operatorList[successor];
                }
                operatorList.erase(op->first);
                break;
            }
        }
    }
    return result;
}

} // namespace infini