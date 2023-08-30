#include "core/api.h"

int main() {
  using namespace infini;
  Data* a = new Data();
  Data* b = new Data();
  Node* add = new Node({a, b});
  Data* temp = add->getOutput(0);
  Node* mul = new Node({b, temp});
  Data* d = mul->getOutput(0);

  add->printInformation();
  mul->printInformation();
  a->printInformation();
  b->printInformation();
  temp->printInformation();
  d->printInformation();

  Graph* g = new Graph({add, mul}, {a, b}, {d});
  g->printInformation();

  delete a;
  delete b;
  delete temp;
  delete d;

  delete add;
  delete mul;

  delete g;

  return 0;
}
