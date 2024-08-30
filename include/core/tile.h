#ifndef TILE_H
#define TILE_H

#include "core/common.h"
#include "core/tensor.h"

namespace infini {

class Tensor;

class Tile {
  public:
    // Tensor Information
    Tensor *tensor;

    // Tile Information
    std::string tileName;
    uint64_t tileOffset;
    Shape tileStartPoint;
    Shape tileCoordinates;

    Shape tileShape;
    Shape tileStride;

    // Codegen Information
    std::vector<std::string> tileCoordsExpr;

  public:
    Tile() = delete;
    Tile(Tensor *tensor, const Shape &shape, const Shape &stride,
         const Shape &coordinates, const uint64_t &offset,
         const std::string &name);
    ~Tile() = default;

    std::string info(bool print = true);

    int64_t getElementNum();
    int64_t getSizeInBytes();

    // For codegen
    std::vector<std::string> tileId2TileCoords(std::string tileId);
    std::string getOffsetInTensor(std::vector<std::string> tileCoords);
    std::string getOffsetInTensor(std::string tileId);
    // Cuda only
    std::string getElementOffsetInTile(std::string threadId);
    std::string getElementOffsetInTensor(std::string blockId,
                                         std::string threadId);
    std::string getElementOffsetInTensor(std::vector<std::string> tileCoords,
                                         std::string threadId);
};

} // namespace infini

#endif
