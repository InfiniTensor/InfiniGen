#include "core/api.h"

int main() {
  using namespace infini;
  Data* a = new Data({1, 2050});
  Data* b = new Data({1, 2050});
  Node* eq = new EQ({a, b});
  Data* temp = eq->getOutput(0);
  Node* div = new SIGMOID({temp});
  Data* d = div->getOutput(0);

  eq->printNode();
  div->printNode();

  a->printData();
  b->printData();
  temp->printData();
  d->printData();

  Graph* graph = new BinaryUnaryGraph({eq, div}, {a, b}, {d});
  graph->printGraph();
  LOG(INFO) << "========== Topo Sort ==========";
  auto topo = graph->topoSort();
  for (auto op : topo) {
    op->printLink();
  }
  LOG(INFO) << "========== Codegen ==========";
  std::string source_code;
  std::string head_code;
  graph->applyPlatform(Platform::BANG);
  LOG(INFO) << "sadfsad";
  source_code = graph->generatorSourceFile();
  head_code = graph->generatorHeadFile();
  LOG_FILE("../code/test.mlu") << source_code;
  LOG_FILE("../binary/test.h") << head_code;
  COMPILE("../code/test.mlu", "../binary/", Platform::BANG);

  delete a;
  delete b;
  delete temp;
  delete d;
  delete eq;
  delete div;
  delete graph;
  return 0;
}
