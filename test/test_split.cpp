#include "core/api.h"

int main() {
  infini::Tensor tensor({1, 1, 6, 6}, infini::TensorDatatype::FLOAT,
                        infini::TensorType::CONST, infini::TensorLayout::NCHW,
                        "abc", 1);
  tensor.printInformation();
  infini::Split split1({1, 1, 2, 2});
  split1.printInformation();
  std::vector<infini::Tile> tilelist = tensor.tiling(split1);

  for (auto i : tilelist) {
    i.printInformation();
  }
  LOG(INFO) << "===============================";
  for (auto i : tilelist) {
    i.printSummary();
  }
  return 0;
}
