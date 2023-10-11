#include "core/api.h"

int main()
{
    using namespace infini;
    std::vector<int64_t> shape = {224, 768};
    Data *a = new Data(shape);
    Data *b = new Data(shape);
    Data *c = new Data(shape);
    Data *d = new Data(shape);
    Node *add = new Node({a, b});
    Data *temp1 = add->getOutput(0);
    Node *sub = new Node({temp1, c});
    Data *temp2 = sub->getOutput(0);
    Node *sqrt = new Node({temp2});
    Data *temp3 = sqrt->getOutput(0);
    Node *mul = new Node({d, temp3});
    Data *temp4 = mul->getOutput(0);
    Node *sub = new Node({temp1, c});
    Data *temp2 = sub->getOutput(0);
    Node *sub = new Node({temp1, c});
    Data *temp2 = sub->getOutput(0);
    Data *d = mul->getOutput(0);

    add->printNode();
    mul->printNode();

    a->printData();
    b->printData();
    temp->printData();
    d->printData();

    Graph *graph = new BinaryUnaryGraph({add, mul}, {a, b}, {d});
    graph->printGraph();
    LOG(INFO) << "========== Topo Sort ==========";
    auto topo = graph->topoSort();
    for (auto op : topo)
    {
        op->printLink();
    }
    LOG(INFO) << "========== Codegen ==========";
    std::string source_code;
    std::string head_code;
    graph->applyPlatform(Platform::CUDA);
    source_code = graph->generatorSourceFile();
    head_code = graph->generatorHeadFile();
    LOG_FILE("../code/test.cu") << source_code;
    LOG_FILE("../binary/test.h") << head_code;
    COMPILE("../code/test.cu", "../binary/", Platform::CUDA);

    delete a;
    delete b;
    delete temp;
    delete d;
    delete add;
    delete mul;
    delete graph;
    return 0;
}
