#include "core/graph.h"
#include "core/utils.h"
#include <algorithm>
#include <unordered_map>

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
    it->remaining += 1;
    if (it->producer != NULL) {
      predecessors.push_back(it->producer);
      it->producer->successors.push_back(this);
    }
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
  info_string += "Predecessors: [";
  for (auto i = 0; i < predecessors.size(); ++i) {
    info_string += predecessors[i]->name;
    info_string += (i == (predecessors.size() - 1) ? "" : ", ");
  }
  info_string += "] ";
  info_string += "Successors: [";
  for (auto i = 0; i < successors.size(); ++i) {
    info_string += successors[i]->name;
    info_string += (i == (successors.size() - 1) ? "" : ", ");
  }
  info_string += "] ";
  LOG(INFO) << info_string;
}

//////////////////////////////////////////////////////////////////////

Data::Data(std::string name_value)
    : name(name_value), index(count++), producer(NULL), remaining(0) {
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
  info_string += "Remaining: [";
  info_string += std::to_string(remaining);
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
      outputs(outputs_list),
      worker_num(1),
      cache_info(1, 1, 1, "", MemoryDispatch::FIFO) {
  if (name == "") {
    name = "Graph_" + std::to_string(index);
  }
  for (auto op : operators) {
    for (auto data : op->inputs) {
      remaining_data.insert(data);  // 不确定是否应该check是否存在
    }
    for (auto data : op->outputs) {
      auto outputs_iter = std::find(outputs.begin(), outputs.end(), data);
      if (outputs_iter == outputs.end()) {
        temps.push_back(data);
      }
    }
  }
}

void Graph::generatorCode() {
  std::vector<Node*> sorted_op = topoSort();
  // Forward pass of graph
  for (auto op : sorted_op) {
    // TODO: codegen
    for (auto input : op->inputs) {
      input->remaining -= 1;
      if (input->remaining == 0) {
        remaining_data.erase(input);
      }
    }
    LOG(INFO) << "Codegen: " + op->name;
    std::string temp = "Remain: [";
    for (auto data : remaining_data) {
      temp += data->name;
      temp += ", ";
    }
    temp += "]";
    LOG(INFO) << temp;
  }
}

void Graph::setDevice(int64_t& worker, Cache& cache) {
  worker_num = worker;
  cache_info = cache;
}

std::vector<Node*> Graph::topoSort() {
  std::unordered_map<Node*, int64_t> operators_temp;
  for (auto op : operators) {
    operators_temp[op] = op->indegree;
  }
  std::vector<Node*> result;
  while (!operators_temp.empty()) {
    for (auto op = operators_temp.begin(); op != operators_temp.end(); ++op) {
      if (op->second == 0) {
        result.push_back(op->first);
        for (auto successor : (op->first)->successors) {
          --operators_temp[successor];
        }
        operators_temp.erase(op->first);
        break;
      }
    }
  }
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
