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

  Graph* graph = new Graph({add, mul}, {a, b}, {e, f, g});
  graph->printInformation();

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

  return 0;
}
