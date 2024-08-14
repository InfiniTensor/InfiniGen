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
    static int64_t tensorCount;

  public:
    // Tensor Information
    std::string tensorName;
    int64_t tensorIndex;
    TensorDataType tensorDataType;
    Shape tensorShape;
    Shape tensorStride;

    // Tiling Information
    Tiles tiles;
    Shape tileGridShape;
    Shape tileGridStride;

    // Graph Information
    int64_t tensorUsesLeft;
    Operator *tensorProducer;
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