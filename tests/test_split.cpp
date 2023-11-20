#include "core/api.h"
#include "core/tensor.h"

int main() {
  infini::Data tensor({1, 1, 7, 7}, infini::TensorDatatype::FLOAT,
                      infini::TensorType::CONST, infini::TensorLayout::NCHW, 1,
                      "abc");
  LOG(INFO) << "Test  Tensor.tiling(Split)";
  LOG(INFO) << "Tensor INFO";
  tensor.printData();
  infini::Split split1({1, 1, 2, 2});
  split1.printInformation();
  infini::TileTensor tiletensor = tensor.tiling(split1);
  LOG(INFO) << "TileTensor INFO";
  tiletensor.printInformation();
  LOG(INFO) << "Tile at tensor(0, 0, 0, 0)";
  auto t = tiletensor({0, 0, 0, 0});
  t.printInformation();
  t = tiletensor({0, 0, 1, 1});
  LOG(INFO) << "Tile at tensor(0, 0, 1, 1)";
  t.printInformation();
  LOG(INFO) << "For loop tiletensor.getTiles()";
  for (auto i : tiletensor.getTiles()) {
    i.printInformation();
  }
  LOG(INFO) << "For loop tiletensor.getTiles() with printSummary";
  for (auto i : tiletensor.getTiles()) {
    i.printSummary();
  }

  LOG(INFO) << "Test Tensor.tiling(vector)";
  tiletensor = tensor.tiling({1, 1, 4, 4});
  LOG(INFO) << "TileTensor INFO";
  tiletensor.printInformation();
  LOG(INFO) << "Tile at tensor(0, 0, 0, 0)";
  t = tiletensor({0, 0, 0, 0});
  t.printInformation();
  t = tiletensor({0, 0, 1, 1});
  LOG(INFO) << "Tile at tensor(0, 0, 1, 1)";
  t.printInformation();
  LOG(INFO) << "For loop tiletensor.getTiles()";
  for (auto i : tiletensor.getTiles()) {
    i.printInformation();
  }
  LOG(INFO) << "For loop tiletensor.getTiles() with printSummary";
  for (auto i : tiletensor.getTiles()) {
    i.printSummary();
  }

  LOG(INFO) << "Test Neat";
  LOG(INFO) << std::to_string(tiletensor.isNeat());
  LOG(INFO) << infini::TO_STRING(tiletensor.neatRange());
  infini::Data tensor1({1, 1, 8, 8}, infini::TensorDatatype::FLOAT,
                       infini::TensorType::CONST, infini::TensorLayout::NCHW, 1,
                       "abc");
  tiletensor = tensor1.tiling({1, 1, 2, 2});
  LOG(INFO) << std::to_string(tiletensor.isNeat());
  return 0;
}