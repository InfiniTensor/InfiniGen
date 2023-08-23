#include "core/kernel.h"
#include "core/utils.h"

namespace infini {

Kernel::Kernel(KernelType type) : kernel_type(type) {}

void Kernel::printInformation() {
  std::string info_string = "";
  info_string += "—— Kernel ";
  info_string += "Name: ";
  info_string += TO_STRING(kernel_type);
  LOG(INFO) << info_string;
}

void Kernel::printSummary() {
  std::string info_string = "";
  info_string += "Kernel ";
  info_string += "Name: ";
  info_string += TO_STRING(kernel_type);
  info_string += "\n";
  LOG(PURE) << info_string;
}

}  // namespace infini
