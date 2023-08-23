#pragma once
#include "core/type.h"
#include "core/split.h"
#include "core/tile.h"
#include <vector>

namespace infini {

class Tensor {
 public:
  // Self information
  TensorDatatype tensor_datatype;
  TensorType tensor_type;
  TensorLayout tensor_layout;
  std::vector<int64_t> tensor_dimension;
  std::vector<int64_t> tensor_stride;
  std::string tensor_name;
  int64_t data_offset;
  bool is_contiguous;

 public:
  // Constructor
  Tensor() = delete;
  Tensor(const std::vector<int64_t>& dimension, TensorDatatype dtype,
         TensorType type, TensorLayout layout, std::string name,
         int64_t offset = 0);
  Tensor(const std::vector<int64_t>& dimension,
         const std::vector<int64_t>& stride, TensorDatatype dtype,
         TensorType type, TensorLayout layout, std::string name,
         int64_t offset = 0);
  // Destructor
  ~Tensor() = default;
  // Tiling
  std::vector<Tile> tiling(const Split& split);
  // Information
  void printInformation();
  void printSummary();
  // Get Function;
  bool isContiguous();
  // Easy Funciton;
  void flatten(int64_t start = 0, int64_t end = -1);
};

}  // namespace infini
