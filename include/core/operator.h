#pragma once
#include "core/tensor.h"
#include "core/worker.h"
#include "core/split.h"
#include "core/attribute.h"
#include "core/type.h"
#include "core/utils.h"
#include "core/kernel.h"
#include <vector>
#include <unordered_map>

namespace infini {

class Operator {
 public:
  // Self information
  OperatorType operator_type;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::unordered_map<std::string, Attribute> attributes;
  // Split information
  Split split;
  std::vector<TileTensor> inputs_tiles;
  std::vector<TileTensor> outputs_tiles;
  // Kernel information
  std::vector<Kernel*> kernel_list;
  // Worker information
  std::vector<Worker*> worker_list;

 public:
  // Constructor
  Operator() = delete;
  Operator(OperatorType type, std::vector<Tensor*> inputs_list,
           std::vector<Tensor*> outputs_list);
  // Destructor
  ~Operator() = default;
  // Set split & Apply split
  void setSplit(Split& split_value);
  virtual void applySplit() = 0;
  // Attributes
  void setAttribute(std::string key, Attribute attribute);
  Attribute getAttribute(std::string key);
  void deleteAttribute(std::string key);
  // Kernels
  void pushKernel(Kernel* kernel);
  // Worker
  void getWorker(std::vector<Worker*> workers);
  // Information
  void printInformation();
  void printSummary();
  // Generator
  // virtual std::string generatorBone(PlatformType platform) = 0;

 private:
  virtual bool checkValid() = 0;
};

}  // namespace infini
