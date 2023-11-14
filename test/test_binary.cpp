#include <tests.h>

int main() {
  int result = std::system("nvcc --version");
  if (result == 0) {
    printf("Testing CUDA Binary...\n");
    test_binary(OperatorType::FLOORMOD, Platform::CUDA);
    test_binary(OperatorType::FLOORDIV, Platform::CUDA);
  }

  result = std::system("cncc --version");
  if (result == 0) {
    printf("Testing BANG Binary...\n");
    test_binary(OperatorType::FLOORMOD, Platform::BANG);
    test_binary(OperatorType::FLOORDIV, Platform::BANG);
  }
}
