#include "core/api.h"
#include "core/platform.h"
#include <string>

int main() {
  infini::Platform cuda(infini::Platform::CUDA);
  infini::Platform bang(infini::Platform::BANG);
  LOG(INFO) << "Test Platform Hander";
  LOG(INFO) << "Test equality";
  LOG(INFO) << int(cuda == bang);
  LOG(INFO) << "Test CUDA codegen";
  LOG(INFO) << cuda.deviceFuncDecl("foo");
  LOG(INFO) << cuda.globalFuncDecl("func");
  LOG(INFO) << cuda.taskIdx();
  LOG(INFO) << cuda.taskDim();
  LOG(INFO) << cuda.regDecl("float", "*cache");
  LOG(INFO) << cuda.ldramDecl("half", "array_ldram");
  LOG(INFO) << cuda.shmemDecl("int8", "arr");
  LOG(INFO) << cuda.glmemDecl("double", "*a");

  LOG(INFO) << "Test BANG codegen";
  LOG(INFO) << cuda.deviceFuncDecl("foo");
  LOG(INFO) << cuda.globalFuncDecl("func");
  LOG(INFO) << cuda.taskIdx();
  LOG(INFO) << cuda.taskDim();
  LOG(INFO) << cuda.regDecl("float", "*cache");
  LOG(INFO) << cuda.ldramDecl("half", "array_ldram");
  LOG(INFO) << cuda.shmemDecl("int8", "arr");
  LOG(INFO) << cuda.glmemDecl("double", "*a");
}
