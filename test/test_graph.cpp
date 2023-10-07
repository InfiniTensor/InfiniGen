#include "core/api.h"

int main() {
  using namespace infini;
  Data* a = new Data({1, 2050});
  Data* b = new Data({1, 2050});
  Node* add = new Node({a, b});
  Data* temp = add->getOutput(0);
  Node* mul = new Node({b, temp});
  Data* d = mul->getOutput(0);

  add->printNode();
  mul->printNode();

  a->printData();
  b->printData();
  temp->printData();
  d->printData();

  Graph* graph = new BinaryUnaryGraph({add, mul}, {a, b}, {d});
  graph->printGraph();
  LOG(INFO) << "========== Topo Sort ==========";
  auto topo = graph->topoSort();
  for (auto op : topo) {
    op->printLink();
  }
  LOG(INFO) << "========== Codegen ==========";
  std::string source_code;
  std::string head_code;
  graph->applyPlatform(Platform::CUDA);
  source_code = graph->generatorSourceFile();
  head_code = graph->generatorHeadFile();
  LOG_FILE("../code/test.cu") << source_code;
  LOG_FILE("../code/test.h") << head_code;

  delete a;
  delete b;
  delete temp;
  delete d;
  delete add;
  delete mul;
  delete graph;
  return 0;
}
