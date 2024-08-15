#ifndef TENSOR_H
#define TENSOR_H

#include "core/common.h"
#include "core/operator.h"
#include "core/tile.h"

namespace infini {

class Operator;

class Tile;
using Tiles = std::vector<Tile *>;

class Tensor {
  private:
    // Number of tensors created
    static int64_t tensorCount;

  public:
    /** Tensor Information **/
    // Name of tensor
    std::string tensorName;
    // Index of tensor
    int64_t tensorIndex;
    // Data type of tensor
    TensorDataType tensorDataType;
    // Shape of tensor
    Shape tensorShape;
    // Stride of tensor
    Shape tensorStride;

    /** Tiling Information **/
    // List of tiles
    Tiles tiles;
    // Shape of tile grid
    Shape tileGridShape;
    // Stride of tile grid
    Shape tileGridStride;

    /** Graph Information **/
    // To keep track of number of remaining usages of tensor in graph
    int64_t tensorUsesLeft;
    // Operator that creates this tensor
    Operator *tensorProducer;
    // Operator(s) that make(s) use of this tensor
    std::vector<Operator *> tensorConsumers;

  public:
    Tensor() = delete;
    Tensor(const Shape &shape,
           const TensorDataType &dataType = TensorDataType::FLOAT,
           const std::string &name = "");
    ~Tensor() = default;

    Tiles tiling(const Shape &shape);

    void addConsumer(Operator *consumer);
    void setProducer(Operator *producer);

    std::string info(bool print = true);
    std::string tilesInfo(bool print = true);
};

} // namespace infini

#endif
