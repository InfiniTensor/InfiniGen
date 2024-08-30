#include "core/tile.h"
#include "core/common.h"
#include "core/log.h"
#include "core/utils.h"

namespace infini {

Tile::Tile(Tensor *tensor_, const Shape &tileShape_, const Shape &tileStride_,
           const Shape &tileCoordinates_, const uint64_t &tileOffset_,
           const std::string &tileName_)
    : tensor(tensor_), tileShape(tileShape_), tileStride(tileStride_),
      tileCoordinates(tileCoordinates_), tileOffset(tileOffset_),
      tileName(tileName_) {}

std::string Tile::info(bool print) {
    std::stringstream out;
    out << BRIGHT_BLACK << HIGHLIGHT << "[TILE] " << RESET << tileName
        << ", Shape: " << TO_STRING(tileShape) << ", "
        << "Stride: " << TO_STRING(tileStride) << ", "
        << "Position: " << TO_STRING(tileCoordinates);

    if (print) {
        LOG(INFO) << out.str();
    }
    return out.str();
}

int64_t Tile::getElementNum() { return VECTOR_PRODUCT(tileShape); }

int64_t Tile::getSizeInBytes() {
    return VECTOR_PRODUCT(tileShape) * SIZE_OF(tensor->tensorDataType);
}

std::vector<std::string> Tile::tileId2TileCoords(std::string tileId) {
    std::vector<std::string> result;
    auto tileGridStride = tensor->tileGridStride;
    if (tileGridStride.size() == 3) {
        /* (tileId / tileGridStride[0]),
           (tileId % tileGridStride[0]) / tileGridStride[1],
           ((tileId % tileGridStride[0]) % tileGridStride[1]) /
           tileGridStride[2]
        */
        result = {
            "(int)(" + tileId + " / " + std::to_string(tileGridStride[0]) + ")",
            "(int)((" + tileId + " % " + std::to_string(tileGridStride[0]) +
                ") / " + std::to_string(tileGridStride[1]) + ")",
            "((" + tileId + " % " + std::to_string(tileGridStride[0]) + ") % " +
                std::to_string(tileGridStride[1]) + ")"};
    } else if (tileGridStride.size() == 2) {
        /* (tileId / tileGridStride[0]),
           (tileId % tileGridStride[0]) / tileGridStride[1]) */
        result = {
            "(int)(" + tileId + " / " + std::to_string(tileGridStride[0]) + ")",
            "(" + tileId + " % " + std::to_string(tileGridStride[0]) + ")"};
    } else if (tileGridStride.size() == 1) {
        /* tileId */
        result = {tileId};
    } else {
        LOG(ERROR) << "tileShape > 3 not supported.";
    }
    return std::move(result);
}

std::string Tile::getOffsetInTensor(std::vector<std::string> tileCoords) {
    std::vector<std::string> result;
    ASSERT(tileCoords.size() == tileShape.size());
    for (int i = 0; i < tileCoords.size(); i++) {
        result.push_back(tileCoords[i] + " * " + std::to_string(tileShape[i]) +
                         " * " + std::to_string(tensor->tensorStride[i]));
    }
    return "(" + STRING_GATHER(result, " + ") + ")";
}

std::string Tile::getOffsetInTensor(std::string tileId) {
    return getOffsetInTensor(tileId2TileCoords(tileId));
}

// Cuda only
std::string Tile::getElementOffsetInTile(std::string threadId) {
    std::string result = "";
    auto tensorStride = tensor->tensorStride;
    if (tileShape.size() == 3) {
        /* (threadId / tileStride[0]) * tensorStride[0] +
           (threadId % tileStride[0]) / tileStride[1] * tensorStride[1] +
           ((threadId % tileStride[0]) % tileStride[1])
         */
        result += "((int)(" + threadId + " / " + std::to_string(tileStride[0]) +
                  ") * " + std::to_string(tensorStride[0]) + " + " + "(int)((" +
                  threadId + " % " + std::to_string(tileStride[0]) + ") / " +
                  std::to_string(tileStride[1]) + ") * " +
                  std::to_string(tensorStride[1]) + " + ((" + threadId + " % " +
                  std::to_string(tileStride[0]) + ") % " +
                  std::to_string(tileStride[1]) + "))";

    } else if (tileShape.size() == 2) {
        /* (threadId / tileStride[0]) * tensorStride[0] +
           (threadId % tileStride[0])
         */
        result += "((int)(" + threadId + " / " + std::to_string(tileStride[0]) +
                  ") * " + std::to_string(tensorStride[0]) + " + " + "(" +
                  threadId + " % " + std::to_string(tileStride[0]) + "))";

    } else if (tileShape.size() == 1) {
        /* threadId */
        result += threadId;

    } else {
        LOG(ERROR) << "tileShape > 3 not supported.";
    }
    return std::move(result);
}

std::string Tile::getElementOffsetInTensor(std::string blockId,
                                           std::string threadId) {
    return getOffsetInTensor(tileId2TileCoords(blockId)) + " + " +
           getElementOffsetInTile(threadId);
}

std::string Tile::getElementOffsetInTensor(std::vector<std::string> tileCoords,
                                           std::string threadId) {
    return getOffsetInTensor(tileCoords) + " + " +
           getElementOffsetInTile(threadId);
}

} // namespace infini
