#pragma once
#include "core/type.h"
#include <vector>

namespace infini {

class Tile {
 public:
  // Self information
  std::vector<int64_t> tile_dimension;
  std::vector<int64_t> tile_stride;
  std::vector<int64_t> tile_position;
  std::string tile_name;
  int64_t start_offset;

 public:
  // Constructor
  Tile() = delete;
  Tile(const std::vector<int64_t>& dimension,
       const std::vector<int64_t>& position, const std::string& name,
       const int64_t& offset);
  Tile(const std::vector<int64_t>& dimension,
       const std::vector<int64_t>& position, const std::vector<int64_t>& stride,
       const std::string& name, const int64_t& offset);
  // Destructor
  ~Tile() = default;
  // Information
  void printInformation();
  void printSummary();
};

}  // namespace infini
