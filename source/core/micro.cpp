#include "core/micro.h"

namespace infini {

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
