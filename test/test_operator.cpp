#include "core/api.h"

int main() {
  infini::Tensor tensor1({3, 4, 5, 6}, infini::TensorDatatype::FLOAT,
                         infini::TensorType::CONST, infini::TensorLayout::NCHW,
                         "left", 0);
  infini::Tensor tensor2({3, 4, 5, 6}, infini::TensorDatatype::FLOAT,
                         infini::TensorType::CONST, infini::TensorLayout::NCHW,
                         "right", 0);
  infini::Tensor tensor3({3, 4, 5, 6}, infini::TensorDatatype::FLOAT,
                         infini::TensorType::VARIABLE,
                         infini::TensorLayout::NCHW, "output", 0);
  tensor1.printInformation();
  tensor2.printInformation();
  tensor3.printInformation();
  infini::Split split({2, 2, 1, 2});
  split.printInformation();
  infini::ADD add(&tensor1, &tensor2, &tensor3);
  LOG(INFO) << "===============================";
  add.setSplit(split);
  add.applySplit();
  add.printInformation();
  LOG(INFO) << "1 ===============================";
  for (auto i = 0; i < add.inputs_tiles[0].size(); ++i) {
    add.inputs_tiles[0][i].printInformation();
  }
  LOG(INFO) << "2 ===============================";
  for (auto i = 0; i < add.inputs_tiles[1].size(); ++i) {
    add.inputs_tiles[1][i].printInformation();
  }
  LOG(INFO) << "3 ===============================";
  for (auto i = 0; i < add.outputs_tiles[0].size(); ++i) {
    add.outputs_tiles[0][i].printInformation();
  }
  LOG(INFO) << "4 ===============================";
  tensor1.flatten();
  tensor2.flatten();
  tensor3.flatten();
  infini::Split spli2({4});
  add.setSplit(spli2);
  add.applySplit();
  add.printInformation();
  LOG(INFO) << "5 ===============================";
  for (auto i = 0; i < add.inputs_tiles[0].size(); ++i) {
    add.inputs_tiles[0][i].printInformation();
  }
  LOG(INFO) << "6 ===============================";
  for (auto i = 0; i < add.inputs_tiles[1].size(); ++i) {
    add.inputs_tiles[1][i].printInformation();
  }
  LOG(INFO) << "7 ===============================";
  for (auto i = 0; i < add.outputs_tiles[0].size(); ++i) {
    add.outputs_tiles[0][i].printInformation();
  }
  return 0;
}
