#include "core/api.h"

int main() {
    using namespace infini;
    std::vector<int64_t> shape = {1, 8, 16};
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
    Operator *mul = new MUL({temp2, d});
    Tensor *temp3 = mul->getOutput(0);
    Operator *bcast = new BROADCAST({temp3}, {}, {4, 8, 16});
    Tensor *temp4 = bcast->getOutput(0);
    Operator *sqrt = new SQRT({temp4});
    Tensor *output = sqrt->getOutput(0);

    add->info();
    sub->info();
    mul->info();
    bcast->info();
    sqrt->info();

    Graph *graph =
        new Graph({add, sub, mul, bcast, sqrt}, {a, b, c, d}, {output});
    graph->info();

    Generator *generator = new Generator(Platform::CUDA, graph, {1, 4, 4});
    LOG(INFO) << generator->generateHeaderFile("build/code/test.h");
    LOG(INFO) << generator->generateSourceFile("build/code/test.cu");
    COMPILE("build/code/test.cu", "build/bin/", Platform::CUDA);

#ifdef DEBUG_MODE
    // Use template for elementwise at the moment 
    generator->generateTestScript(
        "scripts/check/check_elementwise_gpu.template",
        "sqrtf((0[i % 128] + 1[i % 128] - 2[i % 128]) * 3[i % 128])",
        "build/bin/test.cpp");
#endif
    delete a;
    delete b;
    delete c;
    delete d;
    delete temp1;
    delete temp2;
    delete temp3;
    delete temp4;
    delete output;
    delete add;
    delete sub;
    delete mul;
    delete bcast;
    delete sqrt;
    delete graph;
    delete generator;
    return 0;
}
