#include "core/api.h"

int main() {
  infini::Worker worker1(4, 8, 1, "zhangsan", infini::MemoryDispatch::FIFO);
  std::string head = worker1.generatorBoneOnBANG("__nram__", 0);
  std::cout << head << std::endl;
  std::string result = worker1.loadData("a_3_2");
  std::cout << result << std::endl;
  return 0;
}
