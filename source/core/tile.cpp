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

} // namespace infini