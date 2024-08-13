#ifndef TILE_H
#define TILE_H

#include "core/common.h"
#include "core/tensor.h"

namespace infini {

class Tensor;

class Tile {
    // Tensor Information
    Tensor *tensor;

    std::string name;
    uint64_t offset;
    Shape startPoint;
    Shape coordinates;

    Shape shape;
    Shape stride;

  public:
    Tile() = delete;
    Tile(Tensor *tensor, const Shape &shape, const Shape &stride,
         const Shape &coordinates, const uint64_t &offset,
         const std::string &name);
    ~Tile() = default;

    std::string info(bool print = true);
};

} // namespace infini

#endif