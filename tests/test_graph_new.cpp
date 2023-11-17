#include "core/api.h"

int main() {
  using namespace infini;
  std::vector<int64_t> shape = {224, 768};
  Data *a = new Data(shape);
  Data *b = new Data(shape);
  Data *c = new Data(shape);
  Data *d = new Data(shape);
  Node *add = new ADD({a, b});
  Data *temp1 = add->getOutput(0);
  Node *sub = new SUB({temp1, c});
  Data *temp2 = sub->getOutput(0);
  Node *sqrt = new SQRT({temp2});
  Data *temp3 = sqrt->getOutput(0);
  Node *mul = new MUL({d, temp3});
  Data *temp4 = mul->getOutput(0);
  Node *sigmoid = new SIGMOID({temp4});
  Data *output = sigmoid->getOutput(0);

  Graph *graph = new BinaryUnaryGraph({add, sub, sqrt, mul, sigmoid},
                                      {a, b, c, d}, {output});
  LOG(INFO) << "========== Codegen ==========";
  std::string source_code;
  std::string head_code;
  graph->applyPlatform(Platform::BANG);
  source_code = graph->generatorSourceFile();
  head_code = graph->generatorHeadFile();
  LOG_FILE("build/code/test_graph_new.mlu") << source_code;
  LOG_FILE("build/binary/test.h") << head_code;
  COMPILE("build/code/test_graph_new.mlu", "build/bin/", Platform::BANG);

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
