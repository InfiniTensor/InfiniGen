#include "kernels/binary.h"

namespace infini {

BinaryKernel::BinaryKernel() : Kernel(KernelType::BINARY) {}

std::string ADDKernel::generatorCodeOnCUDA(std::vector<std::string> args) {
  return " + ";
}

std::string SUBKernel::generatorCodeOnCUDA(std::vector<std::string> args) {
  return " - ";
}

std::string MULKernel::generatorCodeOnCUDA(std::vector<std::string> args) {
  return " * ";
}

std::string ADDKernel::generatorCodeOnBANG(std::vector<std::string> args) {
  std::string temp = "";
  temp += "__bang_add(";
  temp += args[2] + ", ";
  temp += args[0] + ", ";
  temp += args[1] + ", ";
  temp += args[3] + ");";
  return temp;
}

std::string SUBKernel::generatorCodeOnBANG(std::vector<std::string> args) {
  std::string temp = "";
  temp += "__bang_sub(";
  temp += args[2] + ", ";
  temp += args[0] + ", ";
  temp += args[1] + ", ";
  temp += args[3] + ");";
  return temp;
}

std::string MULKernel::generatorCodeOnBANG(std::vector<std::string> args) {
  std::string temp = "";
  temp += "__bang_mul(";
  temp += args[2] + ", ";
  temp += args[0] + ", ";
  temp += args[1] + ", ";
  temp += args[3] + ");";
  return temp;
}

}  // namespace infini
