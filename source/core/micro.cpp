#include "core/micro.h"

namespace infini {

std::string Micro::printInformation() {
    std::string info_string = "- Kernel ";
    info_string += "KernelType: ";
    info_string += TO_STRING(kernel_type);
    info_string += "; ";
    info_string += "Platform: ";
    info_string += TO_STRING(platform);
    LOG(INFO) << info_string;
}

}  // namespace infini
