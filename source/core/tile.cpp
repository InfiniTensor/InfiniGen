#include "core/tile.h"
#include "core/utils.h"

namespace infini {

// Tile implemenation
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

// TileTensor implementation
TileTensor::TileTensor(const std::vector<int64_t>& shape,
                       const std::vector<int64_t>& stride, TensorType type,
                       TensorLayout layout, std::string name)
    : shape(shape), stride(stride), type(type), layout(layout), name(name) {
  tiles.clear();
}

void TileTensor::addTile(const Tile& t) { tiles.push_back(t); }

Tile TileTensor::deleteTile(const std::vector<int64_t>& coord) {
  int64_t tile_index = DOT_PRODUCT(stride, coord);
  Tile temp = tiles[tile_index];
  tiles.erase(tiles.begin() + tile_index);
  return temp;
}

Tile TileTensor::operator()(const std::vector<int64_t>& coord) {
  // Get a Tile with Tile coord
  // Tile index = stride dot coord
  int64_t tile_index = DOT_PRODUCT(stride, coord);
  return tiles[tile_index];
}

void TileTensor::clear() { tiles.clear(); }

std::vector<Tile> TileTensor::getTiles() { return tiles; }

bool TileTensor::empty() { return tiles.empty(); }

int64_t TileTensor::numTiles() { return tiles.size(); }

int64_t TileTensor::numNeatTiles() {
  return tiles.size() - remain_tiles.size();
}

int64_t TileTensor::numRemainTiles() { return remain_tiles.size(); }

bool TileTensor::isNeat() {
  if (tiles.empty()) {
    return true;
  }
  for (size_t i = 1; i < tiles.size(); i++) {
    if (ANY(tiles[i].tile_dimension != tiles[i - 1].tile_dimension) ||
        ANY(tiles[i].tile_stride != tiles[i - 1].tile_stride)) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> TileTensor::neatRange() {
  std::vector<int64_t> res(shape.size(), 0);
  for (size_t axis = 0; axis < shape.size(); axis++) {
    std::vector<int64_t> coord(shape.size(), 0);
    std::vector<int64_t> coord_pre(shape.size(), 0);
    for (size_t i = 1; i < shape[axis]; i++) {
      coord[axis] = i;
      coord_pre[axis] = i - 1;
      if (ALL((*this)(coord).tile_dimension ==
              (*this)(coord_pre).tile_dimension) &&
          ALL((*this)(coord).tile_stride == (*this)(coord_pre).tile_stride)) {
        continue;
      }
      res[axis] = i;
    }
  }
  return std::move(res);
}

void TileTensor::printInformation() {
  std::string info_string = "";
  info_string += "—— TileTensor ";
  info_string += "Name: ";
  info_string += name;
  info_string += " ";
  info_string += "Shape: ";
  info_string += TO_STRING(shape);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(stride);
  info_string += " ";
  info_string += "TensorType: ";
  info_string += TO_STRING(type);
  info_string += " ";
  info_string += "TensorLayout: ";
  info_string += TO_STRING(layout);
  info_string += " ";
  LOG(INFO) << info_string;
}

void TileTensor::printSummary() {
  std::string info_string = "";
  info_string += "TileTensor ";
  info_string += "Shape: ";
  info_string += TO_STRING(shape);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(stride);
  info_string += " ";
  info_string += "TensorType: ";
  info_string += TO_STRING(type);
  info_string += " ";
  info_string += "TensorLayout: ";
  info_string += TO_STRING(layout);
  info_string += "\n";
  LOG(PURE) << info_string;
}
}  // namespace infini
