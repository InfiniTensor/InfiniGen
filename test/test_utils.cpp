#include "core/api.h"
#include "core/utils.h"

template <class T>
void printVector(const std::vector<T>& v) {
  for (auto i : v) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}

int main() {
  using namespace infini;

  std::vector<int> a = {1, 3, 2, 5};
  std::vector<int> b = {1, 2, 5, 2};
  std::vector<bool> aeqb = (a == b);
  std::vector<bool> abiggerb = (a > b);

  printVector(aeqb);
  printVector(abiggerb);

  std::cout << "ANY abiggerb: " << ANY(abiggerb) << std::endl;
  std::cout << "ALL aeqb: " << ALL(aeqb) << std::endl;
  return 0;
}
