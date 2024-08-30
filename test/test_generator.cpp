#include "core/api.h"

int main() {
    using namespace infini;
    std::vector<int64_t> shape = {8, 32, 16};
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
    Operator *sqrt = new SQRT({temp3});
    Tensor *output = sqrt->getOutput(0);
    add->info();
    sub->info();
    mul->info();
    sqrt->info();

    Graph *graph = new Graph({add, sub, mul, sqrt}, {a, b, c, d}, {output});
    graph->info();

    Generator *generator = new Generator(Platform::CUDA, graph, {4, 8, 8});
    LOG(INFO) << generator->generateHeaderFile("build/code/test.h");
    LOG(INFO) << generator->generateSourceFile("build/code/test.cu");
    COMPILE("build/code/test.cu", "build/bin/", Platform::CUDA);

#ifdef DEBUG_MODE
    generator->generateTestScript(
        "scripts/check/check_elementwise_gpu.template",
        "sqrtf((0[i] + 1[i] - 2[i]) * 3[i])", "build/bin/test.cpp");
#endif
    delete a;
    delete b;
    delete c;
    delete d;
    delete temp1;
    delete temp2;
    delete temp3;
    delete output;
    delete add;
    delete sub;
    delete mul;
    delete sqrt;
    delete graph;
    delete generator;
    return 0;
}
