#pragma
#include "core/kernel.h"

namespace infini{

void Kernel::printInformation(){
    std::string info_string = "- Kernel ";
    info_string += "KernelType: ";
    info_string += TO_STRING(kernel_type);
    info_string += "; ";
    info_string += "Platform: ";
    info_string += TO_STRING(platform);
    LOG(INFO) << info_string;
}

}