#include "core/api.h"

int main() {
  using namespace infini;
  Data* A = new Data({256, 256});
  Data* B = new Data({256, 256});

  Node* gemm = new Gemm({A, B});
  Data* C = gemm->getOutput(0);

  Graph* graph = new GemmGraph({gemm}, {A, B}, {C});

  LOG(INFO) << "========== Codegen ==========";
  std::string source_code;
  std::string head_code;
  graph->applyPlatform(Platform::CUDA);
  source_code = graph->generatorSourceFile();
  head_code = graph->generatorHeadFile();
  LOG_FILE("build/code/test_gemm.cu") << source_code;
  LOG_FILE("build/binary/test.h") << head_code;
  COMPILE("build/code/test_gemm.cu", "build/bin/", Platform::CUDA);
}