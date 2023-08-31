#include "core/graph.h"
#include "core/utils.h"
#include <algorithm>

namespace infini {

int64_t Node::count = 0;
int64_t Data::count = 0;
int64_t Graph::count = 0;

Node::Node(std::vector<Data*> inputs_list, std::vector<Data*> outputs_list,
           std::string name_value, int64_t outputs_num_value)
    : name(name_value),
      index(count++),
      indegree(0),
      outputs_num(outputs_num_value),
      inputs(inputs_list),
      outputs(outputs_list) {
  if (name == "") {
    name = "Operator_" + std::to_string(index);
  }
  if (outputs.empty()) {
    Data* temp;
    for (auto i = 0; i < outputs_num; ++i) {
      temp = new Data();
      outputs.push_back(temp);
    }
  }
  for (auto it : inputs) {
    it->addConsumer(this);
  }
  for (auto it : outputs) {
    it->setProducer(this);
  }
  for (auto it : inputs) {
    indegree += it->producer == NULL ? 0 : 1;
  }
}

Data* Node::getOutput(int64_t index) { return outputs[index]; }

std::vector<Data*> Node::getOutputs() { return outputs; }

void Node::printInformation() {
  std::string info_string = "";
  info_string += "Node ";
  info_string += "Name: [";
  info_string += name;
  info_string += "] ";
  info_string += "Indegree: [";
  info_string += std::to_string(indegree);
  info_string += "] ";
  info_string += "Inputs: [";
  for (auto i = 0; i < inputs.size(); ++i) {
    info_string += inputs[i]->name;
    info_string += (i == (inputs.size() - 1) ? "" : ", ");
  }
  info_string += "] ";
  info_string += "Outputs: [";
  for (auto i = 0; i < outputs.size(); ++i) {
    info_string += outputs[i]->name;
    info_string += (i == (outputs.size() - 1) ? "" : ", ");
  }
  info_string += "] ";
  LOG(INFO) << info_string;
}

//////////////////////////////////////////////////////////////////////

Data::Data(std::string name_value)
    : name(name_value), index(count++), producer(NULL) {
  if (name == "") {
    name = "Data_" + std::to_string(index);
  }
}

void Data::setProducer(Node* producer_value) { producer = producer_value; }

void Data::addConsumer(Node* consumer_value) {
  consumers.push_back(consumer_value);
}

void Data::printInformation() {
  std::string info_string = "";
  info_string += "Data ";
  info_string += "Name: [";
  info_string += name;
  info_string += "] ";
  info_string += "Producer: [";
  info_string += (producer == NULL ? "Null" : producer->name);
  info_string += "] ";
  info_string += "Consumers: [";
  for (auto i = 0; i < consumers.size(); ++i) {
    info_string += consumers[i]->name;
    info_string += (i == (consumers.size() - 1) ? "" : ", ");
  }
  info_string += "] ";
  LOG(INFO) << info_string;
}

//////////////////////////////////////////////////////////////////////

Graph::Graph(std::vector<Node*> operators_list, std::vector<Data*> inputs_list,
             std::vector<Data*> outputs_list, std::string name_value)
    : name(name_value),
      index(count++),
      operators(operators_list),
      inputs(inputs_list),
      outputs(outputs_list) {
  if (name == "") {
    name = "Graph_" + std::to_string(index);
  }
  for (auto op : operators) {
    for (auto data : op->outputs) {
      // auto inputs_iter = std::find(inputs.begin(), inputs.end(), data);
      auto outputs_iter = std::find(outputs.begin(), outputs.end(), data);
      if (outputs_iter == outputs.end()) {
        temps.push_back(data);
      }
    }
  }
}

std::vector<Node*> Graph::topoSort() {
  std::vector<Node*> operators_temp = operators;
  std::vector<Node*> result;
  return result;
}

void Graph::printInformation() {
  std::string info_string = "";
  info_string += "Graph ";
  info_string += "Name: [";
  info_string += name;
  info_string += "] ";
  LOG(INFO) << info_string;
  LOG(INFO) << "==== Operators ====";
  for (auto it : operators) {
    it->printInformation();
  }
  LOG(INFO) << "==== Inputs ====";
  for (auto it : inputs) {
    it->printInformation();
  }
  LOG(INFO) << "==== Temps ====";
  for (auto it : temps) {
    it->printInformation();
  }
  LOG(INFO) << "==== Outputs ====";
  for (auto it : outputs) {
    it->printInformation();
  }
}

}  // namespace infini
