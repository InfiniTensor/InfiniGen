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

    Operator *add = new ADD({a, b});
    Tensor *temp1 = add->getOutput(0);
    Operator *sub = new SUB({temp1, c});
    Tensor *temp2 = sub->getOutput(0);
    Operator *sqrt = new SQRT({temp2});
    Tensor *temp3 = sqrt->getOutput(0);
    Operator *mul = new MUL({d, temp3});
    Tensor *temp4 = mul->getOutput(0);
    Operator *sigmoid = new SIGMOID({temp4});
    Tensor *temp5 = sigmoid->getOutput(0);
    Operator *bcast1 = new BROADCAST({temp5}, {}, {3, 1, 224, 768});
    Tensor *temp6 = bcast1->getOutput(0);
    Tensor *output = new Tensor({3, 3, 224, 768});
    Operator *bcast2 = new BROADCAST({temp6}, {output});

    add->info();
    sub->info();
    sqrt->info();
    mul->info();
    sigmoid->info();
    bcast1->info();
    bcast2->info();

    Graph *graph = new Graph({add, sub, sqrt, mul, sigmoid, bcast1, bcast2},
                             {a, b, c, d}, {output});
    graph->info();

    delete a;
    delete b;
    delete c;
    delete d;
    delete temp1;
    delete temp2;
    delete temp3;
    delete temp4;
    delete temp5;
    delete temp6;
    delete output;
    delete add;
    delete sub;
    delete sqrt;
    delete mul;
    delete sigmoid;
    delete bcast1;
    delete bcast2;
    delete graph;
    return 0;
}
