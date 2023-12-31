#include "core/api.h"
#include "core/tensor.h"

using namespace infini;

int main() {
  LOG(WARNING)
      << "1 ============================================================";
  LOG(INFO) << "Log info success.";
  LOG(WARNING) << "Log warning success.";
  LOG(ERROR) << "Log error success.";
  LOG(FATAL) << "Log fatal success.";

  for (int i = 0; i < 10; ++i) {
    LOG_N(WARNING, 5) << "LOG_N 5 times.";
  }

  infini::Worker worker1(4, 4, 1, "zhangsan", infini::MemoryDispatch::FIFO);
  infini::Worker worker2(4, 4, 1, "lisi", infini::MemoryDispatch::FIFO);
  infini::Worker manger(4, 4, 1, "boss", infini::MemoryDispatch::FIFO);
  infini::Worker slave1(4, 4, 1, "slave", infini::MemoryDispatch::FIFO);
  manger.addSubordinate(&worker1);
  manger.addSubordinate(&worker2);
  worker1.addSubordinate(&slave1);
  manger.printInformation();
  manger.printSummary();
  LOG(WARNING)
      << "2 ============================================================";

  infini::Tensor tensor1({1, 1, 6, 6}, infini::TensorDatatype::FLOAT,
                         infini::TensorType::CONST, infini::TensorLayout::NCHW,
                         "yahaha", 1);
  tensor1.printInformation();
  tensor1.printSummary();
  infini::Split split1({1, 1, 2, 2});
  split1.printInformation();
  infini::Tile tile1({1, 3, 16, 16}, {0, 0, 0, 0}, "hahaya", 0);
  tile1.printInformation();
  tile1.printSummary();

  TileTensor tiletensor = tensor1.tiling(split1);
  tiletensor.printInformation();
  for (auto i : tiletensor.getTiles()) {
    i.printInformation();
  }
  for (auto i : tiletensor.getTiles()) {
    i.printSummary();
  }
  LOG(WARNING)
      << "3 ============================================================";
  DLOG(0) << "DLOG 0";
  DLOG(1) << "DLOG 1";
  DLOG(2) << "DLOG 2";
  DLOG(3) << "DLOG 3";
  DLOG(4) << "DLOG 4";
  return 0;
}
