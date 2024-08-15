#include "core/tensor.h"
#include "core/log.h"
#include "core/utils.h"

namespace infini {

int64_t Tensor::tensorCount = 0;

Tensor::Tensor(const Shape &tensorShape_, const TensorDataType &tensorDataType_,
               const std::string &tensorName_)
    : tensorShape(tensorShape_), tensorDataType(tensorDataType_),
      tensorStride(CALCULATE_STRIDE(tensorShape_)),
      tensorName((tensorName_ == "" ? "Tensor_" + std::to_string(tensorCount)
                                    : tensorName_)),
      tensorIndex(tensorCount++) {}

std::string Tensor::info(bool print) {
    std::stringstream out;
    out << BRIGHT_YELLOW << HIGHLIGHT << "[TENSOR] " << RESET << tensorName
        << TO_STRING(tensorShape) << ", " << TO_STRING(tensorDataType) << ", "
        << "Stride: " << TO_STRING(tensorStride);
    if (print) {
        LOG(INFO) << out.str();
    }
    return out.str();
}

std::string Tensor::tilesInfo(bool print) {
    CHECK(!tiles.empty());
    std::stringstream out;
    for (auto tile : tiles) {
        out << tile->info(print);
    }
    return out.str();
}

Tiles Tensor::tiling(const Shape &pattern) {
    CHECK_EQ(this->tensorShape.size(), pattern.size());
    std::vector<bool> compare = this->tensorShape >= pattern;
    CHECK(ALL_TRUE(compare));
    Shape normalSize = pattern;
    Shape tailSize = this->tensorShape % pattern;
    for (int64_t i = 0; i < tailSize.size(); ++i) {
        tailSize[i] = tailSize[i] == 0 ? normalSize[i] : tailSize[i];
    }
    this->tileGridShape = Shape(pattern.size(), 1);
    for (int64_t i = 0; i < tailSize.size(); ++i) {
        tileGridShape[i] = DIV_UP(this->tensorShape[i], pattern[i]);
    }
    this->tileGridStride = CALCULATE_STRIDE(this->tileGridShape);
    int64_t numTiles = VECTOR_PRODUCT(tileGridShape);
    for (int64_t i = 0; i < numTiles; ++i) {
        int64_t tileIndex = i;
        int64_t axis = 0;
        Shape tileCoordinates;
        while (axis < tileGridStride.size()) {
            tileCoordinates.push_back(tileIndex / tileGridStride[axis]);
            tileIndex %= tileGridStride[axis];
            ++axis;
        }
        Shape tileShape(this->tensorShape.size(), 0);
        for (auto j = 0; j < tileShape.size(); ++j) {
            tileShape[j] =
                (tileCoordinates[j] == (tileGridShape[j] - 1) ? tailSize[j]
                                                              : normalSize[j]);
        }
        Shape tileStride = CALCULATE_STRIDE(tileShape);
        Shape tileStartPoint(this->tensorShape.size(), 0);
        for (auto j = 0; j < tileShape.size(); ++j) {
            tileStartPoint[j] = tileCoordinates[j] * normalSize[j];
        }
        int64_t tileOffset = 0;
        for (auto j = 0; j < this->tensorShape.size(); ++j) {
            tileOffset += tileStartPoint[j] * this->tensorStride[j];
        }
        std::string tileName =
            this->tensorName + "'s Tile " + TO_STRING(tileCoordinates);
        Tile *tile = new Tile(this, tileShape, tileStride, tileCoordinates,
                              tileOffset, tileName);
        this->tiles.push_back(tile);
    }
    return this->tiles;
}

void Tensor::setProducer(Operator *producer) { tensorProducer = producer; }

void Tensor::addConsumer(Operator *consumer) {
    tensorConsumers.push_back(consumer);
}

// bool Tensor::like(const Tensor &other) {
//     std::vector<int64_t> my_dimension = this->tensorShape;
//     std::vector<int64_t> other_dimension = other.tensorShape;
//     size_t my_size = my_dimension.size();
//     size_t other_size = other_dimension.size();
//     if (my_size < other_size) {
//         int pad = other_size - my_size;
//         my_dimension.insert(my_dimension.begin(), pad, 1);
//     } else if (my_size > other_size) {
//         int pad = my_size - other_size;
//         other_dimension.insert(other_dimension.begin(), pad, 1);
//     }
//     if (ALL_TRUE(my_dimension == other_dimension)) {
//         return true;
//     } else {
//         return false;
//     }
// }

} // namespace infini
