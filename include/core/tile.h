#ifndef TILE_H
#define TILE_H

#include "core/common.h"
#include "core/tensor.h"

namespace infini {

class Tensor;

class Tile {
    /** Tensor Information **/
    // Tensor
    Tensor *tensor;

    /** Tile Information */
    // Name of tile
    std::string tileName;
    // Offset of tile relative to tensor in memory
    uint64_t tileOffset;
    // Coordinates of tile's starting point, within tensor
    Shape tileStartPoint;
    // Coordinates of tile, in tile grid
    Shape tileCoordinates;

    // Shape of tile
    Shape tileShape;
    // Stride of tile
    Shape tileStride;

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
