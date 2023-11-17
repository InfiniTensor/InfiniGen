#include <tests.h>

int main() {
  int result = std::system("nvcc --version");
  if (result == 0) {
    printf("Testing CUDA Unary...\n");
    test_unary(OperatorType::COS, Platform::CUDA);
    test_unary(OperatorType::SIN, Platform::CUDA);
    test_unary(OperatorType::TANH, Platform::CUDA);
  }
  result = std::system("cncc --version");
  if (result == 0) {
    printf("Testing BANG Unary...\n");
    test_unary(OperatorType::COS, Platform::BANG);
    test_unary(OperatorType::SIN, Platform::BANG);
    test_unary(OperatorType::TANH, Platform::BANG);
  }
}
