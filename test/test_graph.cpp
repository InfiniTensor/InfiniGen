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
  graph->generatorCode(PlatformType::BANG);

  // LOG(INFO) << "===============================";
  // Data* n_a = new Data({1, 1});
  // Data* n_b = new Data({1, 1});
  // Node* in2out2 = new Node({n_a, n_b}, {}, "", 2);
  // Data* n_e = in2out2->getOutput(0);
  // Data* n_f = in2out2->getOutput(1);
  // Node* in2out1 = new Node({n_e, n_f});
  // Data* n_g = in2out1->getOutput(0);

  // Graph* graph2 = new Graph({in2out2, in2out1}, {n_a, n_b}, {n_g});
  // graph2->printGraph();
  // LOG(INFO) << "========== Topo Sort ==========";
  // topo = graph2->topoSort();
  // for (auto op : topo) {
  //   op->printLink();
  // }
  // LOG(INFO) << "========== Codegen ==========";
  // graph2->generatorCode();

  delete a;
  delete b;
  delete temp;
  delete d;
  delete add;
  delete mul;
  delete graph;

  // delete n_a;
  // delete n_b;
  // delete n_e;
  // delete n_f;
  // delete n_g;
  // delete in2out2;
  // delete in2out1;
  // delete graph2;

  return 0;
}
