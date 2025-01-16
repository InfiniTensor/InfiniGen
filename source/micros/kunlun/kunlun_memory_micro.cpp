#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadKunlun::code(Cache &cache, std::string &code, int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length * dataTypeSize);

    bool cached = (cache.find(operand) != nullptr);
    Block *block = cache.load(operand);
    std::string cachePosStr = "((" + dataTypeStr(dataType) + "*)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + "))";
    if (cached) {
        return cachePosStr;
    }

    std::string tileOffsetStr =
        operand->tensor->tensorName + " + " +
        operand->getOffsetInTensor(operand->tileCoordsExpr);

    if (operand->tileShape.size() == 1) {
        code += INDENTATION(indent) + "GM2LM(" + tileOffsetStr + ", " +
                cachePosStr + ", " + lengthStr + ");\n";
    } else {
        LOG(ERROR) << "\"GM2LM\" only supports 1D.";
    }
    return cachePosStr;
}

std::string StoreKunlun::code(Cache &cache, std::string &code, int64_t indent) {
    std::string lengthStr = std::to_string(length * SIZE_OF(dataType));

    Block *block = cache.load(operand);
    std::string cachePosStr = "((" + dataTypeStr(dataType) + "*)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + "))";
    std::string tileOffsetStr =
        operand->tensor->tensorName + " + " +
        operand->getOffsetInTensor(operand->tileCoordsExpr);

    if (operand->tileShape.size() == 1) {
        code += INDENTATION(indent) + "LM2GM(" + cachePosStr + ", " +
                tileOffsetStr + ", " + lengthStr + ");\n";
    } else {
        LOG(ERROR) << "\"LM2GM\" only supports 1D.";
    }
    return cachePosStr;
}

std::string FreeKunlun::code(Cache &cache, std::string &code, int64_t indent) {
    cache.free(operand);
    return "";
}

std::string AllocateKunlun::code(Cache &cache, std::string &code,
                                 int64_t indent) {
    Block *block = cache.allocate(operand);
    std::string cachePosStr = "((" + dataTypeStr(dataType) + "*)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + "))";
    return cachePosStr;
}

REGISTER_MICRO(OperatorType::LOAD, Platform::KUNLUN, LoadKunlun::makeObj)
REGISTER_MICRO(OperatorType::ALLOCATE, Platform::KUNLUN, LoadKunlun::makeObj)
REGISTER_MICRO(OperatorType::STORE, Platform::KUNLUN, LoadKunlun::makeObj)
REGISTER_MICRO(OperatorType::FREE, Platform::KUNLUN, LoadKunlun::makeObj)

} // namespace infini
