#include "core/graph.h"
#include "core/utils.h"
#include "core/task.h"
#include "micros/binary_micro.h"
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
  name = (name == "" ? "Operator_" + std::to_string(index) : name);
  if (outputs.empty()) {
    Data* temp;
    for (auto i = 0; i < outputs_num; ++i) {
      temp = new Data(inputs[0]->tensor_dimension, inputs[0]->tensor_stride,
                      inputs[0]->tensor_datatype, inputs[0]->tensor_type,
                      inputs[0]->tensor_layout, inputs[0]->data_offset);
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

void Node::printNode() {
  std::string info_string = "";
  info_string += "Operator ";
  info_string += "Name: [";
  info_string += name;
  info_string += "] ";
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

void Node::printLink() {
  std::string info_string = "";
  info_string += "Operator ";
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

void Node::setAttribute(std::string key, Attribute attribute) {
  attributes[key] = attribute;
}

Attribute Node::getAttribute(std::string key) {
  auto iter = attributes.find(key);
  CHECK(iter != attributes.end(), "Can't find this key: " + key);
  return iter->second;
}

void Node::deleteAttribute(std::string key) {
  auto iter = attributes.find(key);
  if (iter != attributes.end()) {
    attributes.erase(iter);
  }
}

//////////////////////////////////////////////////////////////////////

Data::Data(const std::vector<int64_t>& dimension, TensorDatatype dtype,
           TensorType type, TensorLayout layout, int64_t offset,
           std::string name_value)
    : tensor_dimension(dimension),
      tensor_datatype(dtype),
      tensor_type(type),
      tensor_layout(layout),
      data_offset(offset),
      is_contiguous(true),
      name(name_value),
      index(count++),
      producer(NULL),
      remaining(0) {
  name = (name == "" ? "Data_" + std::to_string(index) : name);
  tensor_stride = std::vector<int64_t>(tensor_dimension.size(), 1);
  for (int64_t i = tensor_stride.size() - 2; i >= 0; --i) {
    tensor_stride[i] = tensor_stride[i + 1] * tensor_dimension[i + 1];
  }
}

Data::Data(const std::vector<int64_t>& dimension,
           const std::vector<int64_t>& stride, TensorDatatype dtype,
           TensorType type, TensorLayout layout, int64_t offset,
           std::string name_value)
    : tensor_dimension(dimension),
      tensor_stride(stride),
      tensor_datatype(dtype),
      tensor_type(type),
      tensor_layout(layout),
      data_offset(offset),
      name(name_value),
      index(count++),
      producer(NULL),
      remaining(0) {
  name = (name == "" ? "Data_" + std::to_string(index) : name);
  std::vector<int64_t> temp = std::vector<int64_t>(tensor_dimension.size(), 1);
  for (int64_t i = temp.size() - 2; i >= 0; --i) {
    temp[i] = temp[i + 1] * tensor_dimension[i + 1];
  }
  is_contiguous = ALL(temp == tensor_stride);
}

void Data::setProducer(Node* producer_value) { producer = producer_value; }

void Data::addConsumer(Node* consumer_value) {
  consumers.push_back(consumer_value);
}

void Data::printData() {
  std::string info_string = "";
  info_string += "Tensor ";
  info_string += "Name: [";
  info_string += name;
  info_string += "] ";
  info_string += "Datatype: [";
  info_string += TO_STRING(tensor_datatype);
  info_string += "] ";
  info_string += "Type: [";
  info_string += TO_STRING(tensor_type);
  info_string += "] ";
  info_string += "Layout: [";
  info_string += TO_STRING(tensor_layout);
  info_string += "] ";
  info_string += "Dimension: ";
  info_string += TO_STRING(tensor_dimension);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(tensor_stride);
  info_string += " ";
  info_string += "Offset: [";
  info_string += std::to_string(data_offset);
  info_string += "]";
  LOG(INFO) << info_string;
}

void Data::printLink() {
  std::string info_string = "";
  info_string += "Tensor ";
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

bool Data::isContiguous() { return is_contiguous; }

void Data::flatten(int64_t start, int64_t end) {
  // Check
  int64_t len = tensor_dimension.size();
  CHECK(isContiguous());
  CHECK_GE(start, -len);
  CHECK_LE(start, len - 1);
  CHECK_GE(end, -len);
  CHECK_LE(end, len - 1);
  // Compute
  start = (start + len) % len;
  end = (end + len) % len;
  CHECK_LE(start, end);
  if (start == end) {
    return;
  }
  std::vector<int64_t> result_dimension(len - (end - start), 0);
  for (auto i = 0; i < start; ++i) {
    result_dimension[i] = tensor_dimension[i];
  }
  int64_t accumulate = 1;
  for (auto i = start; i <= end; ++i) {
    accumulate *= tensor_dimension[i];
  }
  result_dimension[start] = accumulate;
  for (auto i = end + 1; i < len; ++i) {
    result_dimension[++start] = tensor_dimension[i];
  }
  // Assign
  tensor_dimension = result_dimension;
  tensor_stride = std::vector<int64_t>(tensor_dimension.size(), 1);
  for (int64_t i = tensor_stride.size() - 2; i >= 0; --i) {
    tensor_stride[i] = tensor_stride[i + 1] * tensor_dimension[i + 1];
  }
}

//////////////////////////////////////////////////////////////////////

Graph::Graph(std::vector<Node*> operators_list, std::vector<Data*> inputs_list,
             std::vector<Data*> outputs_list, std::string name_value)
    : name(name_value),
      index(count++),
      operators(operators_list),
      inputs(inputs_list),
      outputs(outputs_list) {
  name = (name == "" ? "Graph_" + std::to_string(index) : name);
  for (auto op : operators) {
    for (auto data : op->inputs) {
      remaining_data.insert(data);
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
  int64_t tensor_len = VECTOR_PRODUCT(inputs[0]->tensor_dimension);
  int64_t tile_len = 1024;
  int64_t loop = tensor_len / tile_len;
  int64_t rem_len = tensor_len % tile_len;
  std::vector<Node*> sorted_op = topoSort();
  for(int i = 0; i < loop; ++i) {
    LOG(INFO) << "======== Loop =========" << i;
    Task task(1024*10, 1024*100, 128,"__nram__");
    std::unordered_map<Data*, int64_t> temp_remain; 
    for(auto data : inputs) {
      temp_remain[data] = data->remaining;
    }
    for(auto data : temps) {
      temp_remain[data] = data->remaining;
    }
    for(auto data : outputs) {
      temp_remain[data] = data->remaining;
    }
    for (auto op : sorted_op) {
      // TODO: codegen
      for (auto input : op->inputs) {
        temp_remain[input] -= 1;
        if (temp_remain[input] == 0) {
          temp_remain.erase(input);
        }
      }
      BangAddMicro *micro = new BangAddMicro(op->outputs[0]->name,i * tile_len, op->inputs[0]->name, i * tile_len, op->inputs[1]->name, i * tile_len, tile_len);
      task.pushMicro(micro);
      LOG(INFO) << "Codegen: " + op->name;
      std::string temp = "Remain: [";
      for (auto data : temp_remain) {
        temp += data.first->name;
        temp += ", ";
      }
      temp += "]";
      LOG(INFO) << temp;
    }
    LOG(WARNING) << task.generatorCode();
  }
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

void Graph::printGraph() {
  std::string info_string = "";
  info_string += "Graph ";
  info_string += "Name: [";
  info_string += name;
  info_string += "] ";
  LOG(INFO) << info_string;
  LOG(INFO) << "==== Operators ====";
  for (auto it : operators) {
    it->printLink();
  }
  LOG(INFO) << "==== Inputs ====";
  for (auto it : inputs) {
    it->printLink();
  }
  LOG(INFO) << "==== Temps ====";
  for (auto it : temps) {
    it->printLink();
  }
  LOG(INFO) << "==== Outputs ====";
  for (auto it : outputs) {
    it->printLink();
  }
}

}  // namespace infini
