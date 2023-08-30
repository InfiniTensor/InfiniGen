#pragma once
#include "core/type.h"
#include <vector>

namespace infini {

class Data;

class Node {
 private:
  static int64_t count;

 public:
  std::string name;
  const int64_t index;
  std::vector<Data*> inputs;
  std::vector<Data*> outputs;

 public:
  // Constructor
  Node(std::vector<Data*> inputs_list = {},
       std::vector<Data*> outputs_list = {}, std::string name_value = "");
  // Destructor
  ~Node() = default;
  // Function
  Data* getOutput(int64_t index);
  std::vector<Data*> getOutputs();
  // Information
  void printInformation();
};

class Data {
 private:
  static int64_t count;

 public:
  std::string name;
  const int64_t index;
  Node* producer;
  std::vector<Node*> consumers;

 public:
  // Constructor
  Data(std::string name_value = "");
  // Destructor
  ~Data() = default;
  // Function
  void setProducer(Node* producer_value);
  void addConsumer(Node* consumer_value);
  // Information
  void printInformation();
};

class Graph {
 private:
  static int64_t count;

 public:
  std::string name;
  const int64_t index;
  std::vector<Node*> operators;
  std::vector<Data*> inputs;
  std::vector<Data*> outputs;

 public:
  // Constructor
  Graph(std::vector<Node*> operators_list = {},
        std::vector<Data*> inputs_list = {},
        std::vector<Data*> outputs_list = {}, std::string name_value = "");
  // Destructor
  ~Graph() = default;
  // Function
  void topoSort();
  // Information
  void printInformation();
};

}  // namespace infini
