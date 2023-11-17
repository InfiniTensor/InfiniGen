#include "core/api.h"

int main() {
  // Declare Workers
  std::vector<infini::Worker*> workers;
  for (auto i = 0; i < 16; ++i) {
    infini::Worker* ptr = new infini::Worker(
        3, 384 * 1024, 128, std::to_string(i), infini::MemoryDispatch::FIFO);
    workers.push_back(ptr);
  }
  // Declare inputs and output
  infini::Tensor left({1025, 3, 4, 5, 6}, infini::TensorDatatype::FLOAT,
                      infini::TensorType::CONST, infini::TensorLayout::NCHW,
                      "left", 0);
  infini::Tensor right({1025, 3, 4, 5, 6}, infini::TensorDatatype::FLOAT,
                       infini::TensorType::CONST, infini::TensorLayout::NCHW,
                       "right", 0);
  infini::Tensor output({1025, 3, 4, 5, 6}, infini::TensorDatatype::FLOAT,
                        infini::TensorType::VARIABLE,
                        infini::TensorLayout::NCHW, "output", 0);
  // Flatten the tensor
  left.flatten();
  right.flatten();
  output.flatten();
  // Define Operator
  // infini::MUL sub(&left, &right, &output);
  // // Get Worker
  // sub.getWorker(workers);
  // Declare split, operator and apply split
  infini::Split split({16});
  // sub.setSplit(split);
  // sub.applySplit();
  // LOG(PURE) << sub.generatorBone(infini::PlatformType::BANG);
  for (infini::Worker* ptr : workers) {
    delete ptr;
  }
  return 0;
}
