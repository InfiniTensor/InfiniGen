#include "core/tensor.h"
#include "core/utils.h"
#include "core/log.h"

namespace infini {

int64_t Tensor::tensorCount = 0;

Tensor::Tensor(const Shape &shape_, const TensorDataType &dataType_,
               const std::string &name_)
    : shape(shape_), dataType(dataType_), name(name_), index(tensorCount++) {
    name = (name_ == "" ? "Tensor_" + std::to_string(index) : name_);
    stride = CALCULATE_STRIDE(shape);
}

std::string Tensor::info(bool print) {
    std::stringstream out;
    out << name << TO_STRING(shape) << ", " << TO_STRING(dataType) << ", "
        << "Stride: " << TO_STRING(stride);
    if (print) {
        LOG(INFO) << out.str();
    }
    return out.str();
}

Tiles Tensor::tiling(const Shape &pattern) {
    CHECK_EQ(this->shape.size(), pattern.size());
    std::vector<bool> compare = this->shape >= pattern;
    CHECK(ALL_TRUE(compare));
    Shape normalSize = pattern;
    Shape tailSize = this->shape % pattern;
    for (int64_t i = 0; i < tailSize.size(); ++i) {
        tailSize[i] = tailSize[i] == 0 ? normalSize[i] : tailSize[i];
    }
    this->tileGridShape = Shape(pattern.size(), 1);
    for (int64_t i = 0; i < tailSize.size(); ++i) {
        tileGridShape[i] = DIV_UP(this->shape[i], pattern[i]);
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
        Shape tileShape(this->shape.size(), 0);
        for (auto j = 0; j < tileShape.size(); ++j) {
            tileShape[j] =
                (tileCoordinates[j] == (tileGridShape[j] - 1) ? tailSize[j]
                                                             : normalSize[j]);
        }
        Shape tileStride = CALCULATE_STRIDE(tileShape);
        Shape tileStartPoint(this->shape.size(), 0);
        for (auto j = 0; j < tileShape.size(); ++j) {
            tileStartPoint[j] = tileCoordinates[j] * normalSize[j];
        }
        int64_t tileOffset = 0;
        for (auto j = 0; j < this->shape.size(); ++j) {
            tileOffset += tileStartPoint[j] * this->stride[j];
        }
        std::string tileName =
            this->name + "'s Tile " + TO_STRING(tileCoordinates);
        Tile *tile = new Tile(this, tileShape, tileStride, tileCoordinates,
                              tileOffset, tileName);
        this->tiles.push_back(tile);
    }
    return this->tiles;
}

// void Tensor::setProducer(Operator *producer_value) {
//     tensor_producer = producer_value;
// }

// void Tensor::addConsumer(Operator *consumer_value) {
//     tensor_consumers.push_back(consumer_value);
// }

// bool Tensor::like(const Tensor &other) {
//     std::vector<int64_t> my_dimension = this->shape;
//     std::vector<int64_t> other_dimension = other.shape;
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