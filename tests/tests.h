#include <core/api.h>
#include <core/type.h>

using namespace infini;

void test_unary(OperatorType op_type, Platform plat) {
  Data *a = new Data({100});
  Node *op = nullptr;
  std::string op_name;
  std::string plat_name;
  std::string suffix;
  switch (op_type) {
    case OperatorType::SQRT:
      op = new SQRT({a});
      op_name = "sqrt";
      break;
    case OperatorType::RELU:
      op = new RELU({a});
      op_name = "relu";
      break;
    case OperatorType::SIN:
      op = new SIN({a});
      op_name = "sin";
      break;
    case OperatorType::COS:
      op = new COS({a});
      op_name = "cos";
      break;
    case OperatorType::TANH:
      op = new TANH({a});
      op_name = "tanh";
      break;
    default:
      fprintf(stderr, "Unsupported operator type.\n");
      exit(1);
  }

  if (plat == Platform::CUDA) {
    plat_name = "cuda";
    suffix = "cu";
  } else if (plat == Platform::BANG) {
    plat_name = "bang";
    suffix = "mlu";
  } else {
    fprintf(stderr, "Unsupported platform.\n");
    exit(1);
  }

  Graph *graph = new BinaryUnaryGraph({op}, {a}, {op->getOutput(0)});

  LOG(INFO) << "========== Topo Sort ==========";
  auto topo = graph->topoSort();
  for (auto op : topo) {
    op->printLink();
  }

  LOG(INFO) << "========== Codegen ==========";
  std::string source_code;
  std::string head_code;
  graph->applyPlatform(plat);
  source_code = graph->generatorSourceFile();
  head_code = graph->generatorHeadFile();
  char source_file[50];
  char head_file[50];
  snprintf(source_file, 50, "build/code/test_%s_%s.%s", op_name.c_str(),
           plat_name.c_str(), suffix.c_str());
  snprintf(head_file, 50, "build/bin/test_%s_%s.h", op_name.c_str(),
           plat_name.c_str());
  LOG_FILE(source_file) << source_code;
  LOG_FILE(head_file) << head_code;
  COMPILE(source_file, "build/bin/", plat);
}

void test_binary(OperatorType op_type, Platform plat) {
  Data *a = new Data({100});
  Data *b = new Data({100});
  Node *op = nullptr;
  std::string op_name;
  std::string plat_name;
  std::string suffix;
  switch (op_type) {
    case OperatorType::FLOORMOD:
      op = new FLOORMOD({a, b});
      op_name = "floormod";
      break;
    case OperatorType::FLOORDIV:
      op = new FLOORDIV({a, b});
      op_name = "floordiv";
      break;
    default:
      fprintf(stderr, "Unsupported operator type.\n");
      exit(1);
  }

  if (plat == Platform::CUDA) {
    plat_name = "cuda";
    suffix = "cu";
  } else if (plat == Platform::BANG) {
    plat_name = "bang";
    suffix = "mlu";
  } else {
    fprintf(stderr, "Unsupported platform.\n");
    exit(1);
  }

  Graph *graph = new BinaryUnaryGraph({op}, {a, b}, {op->getOutput(0)});

  LOG(INFO) << "========== Topo Sort ==========";
  auto topo = graph->topoSort();
  for (auto op : topo) {
    op->printLink();
  }

  LOG(INFO) << "========== Codegen ==========";
  std::string source_code;
  std::string head_code;
  graph->applyPlatform(plat);
  source_code = graph->generatorSourceFile();
  head_code = graph->generatorHeadFile();
  char source_file[50];
  char head_file[50];
  snprintf(source_file, 50, "build/code/test_%s_%s.%s", op_name.c_str(),
           plat_name.c_str(), suffix.c_str());
  snprintf(head_file, 50, "build/bin/test_%s_%s.h", op_name.c_str(),
           plat_name.c_str());
  LOG_FILE(source_file) << source_code;
  LOG_FILE(head_file) << head_code;
  COMPILE(source_file, "build/bin/", plat);
}
