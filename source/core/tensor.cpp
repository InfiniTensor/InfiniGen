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
      tensorIndex(tensorCount++), tensorUsesLeft(0), tensorProducer(nullptr) {}

Tensor::~Tensor() {
    for (auto tile : tiles) {
        delete tile;
    }
}

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

int64_t Tensor::getElementNum() { return VECTOR_PRODUCT(tensorShape); }

int64_t Tensor::getSizeInBytes() {
    return VECTOR_PRODUCT(tensorShape) * SIZE_OF(tensorDataType);
}

} // namespace infini
