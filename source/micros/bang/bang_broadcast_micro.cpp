#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/broadcast_micro.h"
#include "micros/memory_micro.h"

namespace infini {

std::string BroadcastBang::code(Cache &cache, std::string &code,
                                int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length * dataTypeSize);

    cache.lock();
    std::string inputCache = LoadBang({input}).code(cache, code, indent);
    std::string outputCache = AllocateBang({output}).code(cache, code, indent);

    code += INDENTATION(indent) + "__memcpy(" + outputCache + ", " +
            inputCache + ", " + lengthStr + ", NRAM2NRAM);\n";
    cache.unlock();
    return "";
}

REGISTER_MICRO(OperatorType::BROADCAST, Platform::BANG, BroadcastBang::makeObj)

} // namespace infini
