#include "core/tile.h"
#include "core/utils.h"

namespace infini {

Tile::Tile(const std::vector<int64_t>& dimension,
           const std::vector<int64_t>& position, const std::string& name,
           const int64_t& offset) {
  tile_dimension = dimension;
  tile_position = position;
  tile_name = name;
  start_offset = offset;
  tile_stride = std::vector<int64_t>(tile_dimension.size(), 1);
  for (int64_t i = tile_stride.size() - 2; i >= 0; --i) {
    tile_stride[i] = tile_stride[i + 1] * tile_dimension[i + 1];
  }
}

Tile::Tile(const std::vector<int64_t>& dimension,
           const std::vector<int64_t>& position,
           const std::vector<int64_t>& stride, const std::string& name,
           const int64_t& offset) {
  tile_dimension = dimension;
  tile_position = position;
  tile_stride = stride;
  tile_name = name;
  start_offset = offset;
}

void Tile::printInformation() {
  std::string info_string = "";
  info_string += "—— Tile ";
  info_string += "Name: ";
  info_string += tile_name;
  info_string += " ";
  info_string += "Dimension: ";
  info_string += TO_STRING(tile_dimension);
  info_string += " ";
  info_string += "Position: ";
  info_string += TO_STRING(tile_position);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(tile_stride);
  info_string += " ";
  info_string += "Offset: ";
  info_string += std::to_string(start_offset);
  LOG(INFO) << info_string;
}

void Tile::printSummary() {
  std::string info_string = "";
  info_string += "Tile ";
  info_string += "Dim: ";
  info_string += TO_STRING(tile_dimension);
  info_string += " ";
  info_string += "Pos: ";
  info_string += TO_STRING(tile_position);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(tile_stride);
  info_string += " ";
  info_string += "Offset: ";
  info_string += std::to_string(start_offset);
  info_string += "\n";
  LOG(PURE) << info_string;
}

}  // namespace infini
