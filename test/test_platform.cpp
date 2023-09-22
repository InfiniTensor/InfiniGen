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
  LOG(INFO)<<cuda.deviceFuncDecl();
  LOG(INFO)<<cuda.globalFuncDecl();
  LOG(INFO)<<cuda.taskIdxDecl();
  LOG(INFO)<<cuda.taskDimDecl();
  LOG(INFO)<<cuda.regDecl();
  LOG(INFO)<<cuda.ldramDecl();
  LOG(INFO)<<cuda.shmemDecl();
  LOG(INFO)<<cuda.glmemDecl();

  LOG(INFO) << "Test BANG codegen";
  LOG(INFO)<<bang.deviceFuncDecl();
  LOG(INFO)<<bang.globalFuncDecl();
  LOG(INFO)<<bang.taskIdxDecl();
  LOG(INFO)<<bang.taskDimDecl();
  LOG(INFO)<<bang.regDecl();
  LOG(INFO)<<bang.ldramDecl();
  LOG(INFO)<<bang.shmemDecl();
  LOG(INFO)<<bang.glmemDecl();
}