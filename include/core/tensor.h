#ifndef TENSOR_H
#define TENSOR_H

#include "core/common.h"
#include "core/tile.h"

namespace infini {

class Tile;
using Tiles = std::vector<Tile *>;

class Tensor {
  private:
    static int64_t tensorCount;

  public:  
    // Tensor Information
    std::string name;
    TensorDataType dataType;
    Shape shape;
    Shape stride;
    int64_t index;

    // Tiling Information
    Tiles tiles;
    Shape tileGridShape;
    Shape tileGridStride;

    // Graph Information

  public:
    Tensor() = delete;
    Tensor(const Shape &shape,
           const TensorDataType &dataType = TensorDataType::FLOAT,
           const std::string &name = "");
    ~Tensor() = default;

    Tiles tiling(const Shape &shape);

    std::string info(bool print = true);
};

} // namespace infini

#endif