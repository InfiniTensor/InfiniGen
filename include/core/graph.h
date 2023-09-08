#pragma once
#include "core/type.h"
#include "core/cache.h"
#include <vector>
#include <unordered_set>

namespace infini {

class Data;

class Node {
 private:
  static int64_t count;

 public:
  std::string name;
  const int64_t index;
  int64_t indegree;
  int64_t outputs_num;
  std::vector<Data*> inputs;
  std::vector<Data*> outputs;
  std::vector<Node*> predecessors;
  std::vector<Node*> successors;

 public:
  // Constructor
  Node(std::vector<Data*> inputs_list = {},
       std::vector<Data*> outputs_list = {}, std::string name_value = "",
       int64_t outputs_num_value = 1);
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
  int remaining;
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
  std::vector<Data*> temps;
  std::unordered_set<Data*> remaining_data;

  Cache cache_info;
  int64_t worker_num;

 public:
  // Constructor
  Graph(std::vector<Node*> operators_list = {},
        std::vector<Data*> inputs_list = {},
        std::vector<Data*> outputs_list = {}, std::string name_value = "");
  // Destructor
  ~Graph() = default;
  // Function
  std::vector<Node*> topoSort();
  virtual void generatorCode();
  void setDevice(int64_t& worker, Cache& cache);
  // Information
  void printInformation();
};

}  // namespace infini
