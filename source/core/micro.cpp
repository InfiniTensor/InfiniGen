#include "core/micro.h"
#include "core/utils.h"

namespace infini {

Micro::Micro(MicroType mt, PlatformType pt) : micro_type(mt), platform(pt) {
  switch (platform) {
    case PlatformType::CUDA:
      coreIndex = "blockIdx.x";
      break;
    case PlatformType::BANG:
      coreIndex = "taskId";
      break;
    default:
      coreIndex = "";
      break;
  }
}

void Micro::printInformation() {
  std::string info_string = "- Kernel ";
  info_string += "KernelType: ";
  info_string += TO_STRING(micro_type);
  info_string += "; ";
  info_string += "Platform: ";
  info_string += TO_STRING(platform);
  LOG(INFO) << info_string;
}

}  // namespace infini
