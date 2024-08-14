#include "core/api.h"

int main() {
    using namespace infini;
    std::vector<int64_t> shape = {224, 768};
    Tensor *a = new Tensor(shape);
    Tensor *b = new Tensor(shape);
    Tensor *c = new Tensor(shape);
    Tensor *d = new Tensor(shape);
    a->info();
    b->info();
    c->info();
    d->info();

    Operator *add = new Operator({a, b});
    Tensor *temp1 = add->getOutput(0);
    Operator *sub = new Operator({temp1, c});
    Tensor *temp2 = sub->getOutput(0);
    Operator *sqrt = new Operator({temp2});
    Tensor *temp3 = sqrt->getOutput(0);
    Operator *mul = new Operator({d, temp3});
    Tensor *temp4 = mul->getOutput(0);
    Operator *sigmoid = new Operator({temp4});
    Tensor *output = sigmoid->getOutput(0);
    add->info();
    sub->info();
    sqrt->info();
    mul->info();
    sigmoid->info();

    Graph *graph =
        new Graph({add, sub, sqrt, mul, sigmoid}, {a, b, c, d}, {output});
    graph->info();

    delete a;
    delete b;
    delete c;
    delete d;
    delete add;
    delete sub;
    delete sqrt;
    delete mul;
    delete sigmoid;
    delete output;
    delete graph;
    return 0;
}