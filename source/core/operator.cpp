#include "core/operator.h"

namespace infini {

Operator::Operator(OperatorType type, std::vector<Tensor*> inputs_list,
                   std::vector<Tensor*> outputs_list)
    : operator_type(type), inputs(inputs_list), outputs(outputs_list) {}

void Operator::setSplit(Split& split_value) { split = split_value; }

void Operator::setAttribute(std::string key, Attribute attribute) {
  attributes[key] = attribute;
}

Attribute Operator::getAttribute(std::string key) {
  auto iter = attributes.find(key);
  CHECK(iter != attributes.end(), "Can't find this key: " + key);
  return iter->second;
}

void Operator::deleteAttribute(std::string key) {
  auto iter = attributes.find(key);
  if (iter != attributes.end()) {
    attributes.erase(iter);
  }
}

void Operator::pushKernel(Kernel* kernel) { kernel_list.push_back(kernel); }

void Operator::getWorker(std::vector<Worker*> workers) {
  worker_list = workers;
}

void Operator::printInformation() {
  std::string info_string = "";
  info_string += "—— Operator ";
  info_string += "Name: ";
  info_string += TO_STRING(operator_type);
  info_string += " ";
  info_string += "Inputs: [";
  for (auto i = 0; i < inputs.size(); ++i) {
    info_string += TO_STRING(inputs[i]->tensor_dimension);
    info_string += (i == (inputs.size() - 1) ? "" : ",");
  }
  info_string += "]";
  info_string += " ";
  info_string += "Outputs: [";
  for (auto i = 0; i < outputs.size(); ++i) {
    info_string += TO_STRING(outputs[i]->tensor_dimension);
    info_string += (i == (outputs.size() - 1) ? "" : ",");
  }
  info_string += "]";
  LOG(INFO) << info_string;
}

void Operator::printSummary() {
  std::string info_string = "";
  info_string += "Operator ";
  info_string += "Inputs: [";
  for (auto i = 0; i < inputs.size(); ++i) {
    info_string += TO_STRING(inputs[i]->tensor_dimension);
    info_string += (i == (inputs.size() - 1) ? "" : ",");
  }
  info_string += "]";
  info_string += " ";
  info_string += "Outputs: [";
  for (auto i = 0; i < outputs.size(); ++i) {
    info_string += TO_STRING(outputs[i]->tensor_dimension);
    info_string += (i == (outputs.size() - 1) ? "" : ",");
  }
  info_string += "]";
  info_string += "\n";
  LOG(PURE) << info_string;
}

}  // namespace infini
