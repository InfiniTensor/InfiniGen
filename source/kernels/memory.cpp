#include "kernels/memory.h"

namespace infini {

MemoryKernel::MemoryKernel() : Kernel(KernelType::MEMORY) {}

std::string G2RKernel::generatorCodeOnCUDA(std::vector<std::string> args) {
  std::string temp = "";
  temp += args[0] + "[threadIdx.y]";
  return temp;
}

std::string R2GKernel::generatorCodeOnCUDA(std::vector<std::string> args) {
  std::string temp = "";
  temp += args[0] + "[threadIdx.y]";
  return temp;
}

std::string G2RKernel::generatorCodeOnBANG(std::vector<std::string> args) {
  std::string temp = "";
  temp += "__memcpy(";
  temp += args[1] + ", ";
  temp += args[0] + ", ";
  temp += args[2] + ", ";
  temp += args[3] + ");";
  return temp;
}

std::string R2GKernel::generatorCodeOnBANG(std::vector<std::string> args) {
  std::string temp = "";
  temp += "__memcpy(";
  temp += args[1] + ", ";
  temp += args[0] + ", ";
  temp += args[2] + ", ";
  temp += args[3] + ");";
  return temp;
}

}  // namespace infini
