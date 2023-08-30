#include "core/graph.h"
#include "core/utils.h"

namespace infini {

int64_t Node::count = 0;
int64_t Data::count = 0;
int64_t Graph::count = 0;

Node::Node(std::vector<Data*> inputs_list, std::vector<Data*> outputs_list,
           std::string name_value)
    : name(name_value),
      index(count++),
      inputs(inputs_list),
      outputs(outputs_list) {
  if (name == "") {
    name = "Operator_" + std::to_string(index);
  }
  if (outputs.empty()) {
    Data* temp = new Data();
    outputs.push_back(temp);
  }
  for (auto it : inputs) {
    it->addConsumer(this);
  }
  for (auto it : outputs) {
    it->setProducer(this);
  }
}

Data* Node::getOutput(int64_t index) { return outputs[index]; }

std::vector<Data*> Node::getOutputs() { return outputs; }

void Node::printInformation() {
  std::string info_string = "";
  info_string += "Node ";
  info_string += "Name: ";
  info_string += name;
  info_string += " ";
  info_string += "Inputs: [";
  for (auto i = 0; i < inputs.size(); ++i) {
    info_string += inputs[i]->name;
    info_string += (i == (inputs.size() - 1) ? "" : ",");
  }
  info_string += "]";
  info_string += " ";
  info_string += "Outputs: [";
  for (auto i = 0; i < outputs.size(); ++i) {
    info_string += outputs[i]->name;
    info_string += (i == (outputs.size() - 1) ? "" : ",");
  }
  info_string += "]";
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
  info_string += "Name: ";
  info_string += name;
  info_string += " ";
  info_string += "Producer: [";
  info_string += (producer == NULL ? "Null" : producer->name);
  info_string += "] ";
  info_string += "Consumers: [";
  for (auto i = 0; i < consumers.size(); ++i) {
    info_string += consumers[i]->name;
    info_string += (i == (consumers.size() - 1) ? "" : ",");
  }
  info_string += "]";
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
}

void Graph::topoSort() {
  // TODO(wanghailu)
}

void Graph::printInformation() {
  std::string info_string = "";
  info_string += "Graph ";
  info_string += "Name: ";
  info_string += name;
  LOG(INFO) << info_string;
  LOG(INFO) << "==== Operators ====";
  for (auto it : operators) {
    it->printInformation();
  }
  LOG(INFO) << "==== Inputs ====";
  for (auto it : inputs) {
    it->printInformation();
  }
  LOG(INFO) << "==== Outputs ====";
  for (auto it : outputs) {
    it->printInformation();
  }
}

}  // namespace infini
