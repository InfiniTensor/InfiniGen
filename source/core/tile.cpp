#include "core/tile.h"
#include "core/common.h"
#include "core/log.h"
#include "core/utils.h"

namespace infini {

Tile::Tile(Tensor *tensor_, const Shape &shape_, const Shape &stride_,
           const Shape &coordinates_, const uint64_t &offset_,
           const std::string &name_)
    : tensor(tensor_), shape(shape_), stride(stride_),
      coordinates(coordinates_), offset(offset_), name(name_) {}

std::string Tile::info(bool print) {
    std::stringstream out;
    out << name << ", Shape: " << TO_STRING(shape) << ", "
        << "Stride: " << TO_STRING(stride) << ", "
        << "Position: " << TO_STRING(coordinates);

    if (print) {
        LOG(INFO) << out.str();
    }
    return out.str();
}

} // namespace infini