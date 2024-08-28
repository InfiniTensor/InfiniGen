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

    Operator *add = new Operator({a, b});
    add->operatorType = OperatorType::ADD;
    Tensor *temp1 = add->getOutput(0);
    Operator *sub = new Operator({temp1, c});
    sub->operatorType = OperatorType::SUB;
    Tensor *temp2 = sub->getOutput(0);
    Operator *mul = new Operator({temp2, d});
    mul->operatorType = OperatorType::MUL;
    Tensor *temp3 = mul->getOutput(0);
    Operator *sqrt = new Operator({temp3});
    sqrt->operatorType = OperatorType::SQRT;
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
        "sqrtf((0 + 1 - 2) * 3)", "build/bin/test.cpp");
#endif
    delete a;
    delete b;
    delete c;
    delete d;
    delete add;
    delete sub;
    delete mul;
    delete output;
    delete graph;
    return 0;
}
