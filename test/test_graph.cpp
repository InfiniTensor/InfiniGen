#include "core/api.h"

int main() {
  using namespace infini;
  Data* a = new Data();
  Data* b = new Data();
  Node* add = new Node({a, b});
  Data* temp = add->getOutput(0);
  Node* mul = new Node({b, temp});
  Data* d = mul->getOutput(0);
  Node* com = new Node({d, temp}, {}, "", 3);
  Data* e = com->getOutput(0);
  Data* f = com->getOutput(1);
  Data* g = com->getOutput(2);

  add->printInformation();
  mul->printInformation();
  com->printInformation();

  a->printInformation();
  b->printInformation();
  temp->printInformation();
  d->printInformation();
  e->printInformation();
  f->printInformation();
  g->printInformation();

  Graph* graph = new BinaryUnaryGraph({add, mul, com}, {a, b}, {e, f, g});
  graph->printInformation();
  LOG(INFO) << "========== Topo Sort ==========";
  auto topo = graph->topoSort();
  for (auto op : topo) {
    op->printInformation();
  }
  LOG(INFO) << "========== Codegen ==========";
  graph->generatorCode();

  LOG(INFO) << "===============================";
  Data* n_a = new Data();
  Data* n_b = new Data();
  Node* in2out2 = new Node({n_a, n_b}, {}, "", 2);
  Data* n_e = in2out2->getOutput(0);
  Data* n_f = in2out2->getOutput(1);
  Node* in2out1 = new Node({n_e, n_f});
  Data* n_g = in2out1->getOutput(0);

  Graph* graph2 = new BinaryUnaryGraph({in2out2, in2out1}, {n_a, n_b}, {n_g});
  graph2->printInformation();
  LOG(INFO) << "========== Topo Sort ==========";
  topo = graph2->topoSort();
  for (auto op : topo) {
    op->printInformation();
  }
  LOG(INFO) << "========== Codegen ==========";
  graph2->generatorCode();

  delete a;
  delete b;
  delete temp;
  delete d;
  delete e;
  delete f;
  delete g;
  delete add;
  delete mul;
  delete com;
  delete graph;

  delete n_a;
  delete n_b;
  delete n_e;
  delete n_f;
  delete n_g;
  delete in2out2;
  delete in2out1;
  delete graph2;

  return 0;
}
