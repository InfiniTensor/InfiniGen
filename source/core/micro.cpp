#include "core/micro.h"
#include "core/utils.h"

namespace infini {

Micro *Micro::makeObj() { return nullptr; }

std::string Micro::info(bool print) {
    std::stringstream out;
    out << CYAN << HIGHLIGHT << "[MICRO] " << RESET << TO_STRING(microType);
    if (print) {
        LOG(INFO) << out.str();
    }
    return out.str();
}

} // namespace infini
