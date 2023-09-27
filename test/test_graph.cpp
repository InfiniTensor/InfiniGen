#include "core/api.h"

int main() {
  using namespace infini;
  Data* a = new Data({2, 1024});
  Data* b = new Data({2, 1024});
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
  std::string code;
  graph->applyPlatform(Platform::BANG);
  code += graph->generatorHead();
  code += graph->generatorTask();
  code += graph->generatorHost();
  code += graph->generatorCode();
  LOG_FILE("test.mlu") << code;

  delete a;
  delete b;
  delete temp;
  delete d;
  delete add;
  delete mul;
  delete graph;
  return 0;
}
