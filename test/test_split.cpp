#include "core/api.h"
#include "core/tensor.h"

int main() {
  infini::Tensor tensor({1, 1, 7, 7}, infini::TensorDatatype::FLOAT,
                        infini::TensorType::CONST, infini::TensorLayout::NCHW,
                        "abc", 1);
  LOG(INFO) << "Tensor INFO";
  tensor.printInformation();
  infini::Split split1({1, 1, 2, 2});
  split1.printInformation();
  infini::TileTensor tiletensor = tensor.tiling(split1);
  LOG(INFO) << "TileTensor INFO";
  tiletensor.printInformation();
  LOG(INFO) << "Tile at tensor(0, 0, 0, 0)";
  auto t = tiletensor({0,0,0,0}); 
  t.printInformation();
  t = tiletensor({0,0,1,1});
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
  return 0;
}
