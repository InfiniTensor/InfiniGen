#include "core/api.h"

int main() {
    infini::Tensor tensor1({3, 4, 5, 6, 7}, infini::TensorDataType::FLOAT);
    infini::Tensor tensor2({4, 5, 6, 7}, infini::TensorDataType::DOUBLE);
    LOG(INFO) << "Tensor 1 Information";
    tensor1.info();
    LOG(INFO) << "Tensor 2 Information";
    LOG(INFO) << tensor2.info(false);

    infini::Tiles tiles2 = tensor2.tiling({2, 2, 3, 5});
    tensor2.tilesInfo();

    return 0;
}
