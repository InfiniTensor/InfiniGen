#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadCuda::code(Cache &cache, std::string &code, int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length * dataTypeSize);

    bool cached = (cache.find(operand) != nullptr);
    Block *block = cache.load(operand);
    std::string cachePosStr =
        "((" + dataTypeStr(dataType) + " *)(" + cache.cacheName + " + " +
        std::to_string(block->blockStart) + "))[" + platform.threadId() + "]";
    if (cached) {
        return cachePosStr;
    }

    // use blockId/threadId to get element offset
    std::string threadOffsetStr =
        operand->tensor->tensorName + "[" +
        platform.offset(operand->tensor->tensorStride,
                        operand->tensor->tileGridStride, operand->tileShape) +
        " + " +
        platform.offset(operand->tensor->tensorStride, operand->tileStride,
                        std::vector<int64_t>(operand->tileShape.size(), 1),
                        true) +
        "]";

    code += INDENTATION(indent) + cachePosStr + " = " + threadOffsetStr + ";\n";
    return cachePosStr;
}

std::string StoreCuda::code(Cache &cache, std::string &code, int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length * dataTypeSize);

    Block *block = cache.load(operand);
    std::string cachePosStr =
        "((" + dataTypeStr(dataType) + " *)(" + cache.cacheName + " + " +
        std::to_string(block->blockStart) + "))[" + platform.threadId() + "]";

    // use blockId/threadId to get element offset
    std::string threadOffsetStr =
        operand->tensor->tensorName + "[" +
        platform.offset(operand->tensor->tensorStride,
                        operand->tensor->tileGridStride, operand->tileShape) +
        " + " +
        platform.offset(operand->tensor->tensorStride, operand->tileStride,
                        std::vector<int64_t>(operand->tileShape.size(), 1),
                        true) +
        "]";

    code += INDENTATION(indent) + threadOffsetStr + " = " + cachePosStr + ";\n";
    return cachePosStr;
}

std::string FreeCuda::code(Cache &cache, std::string &code, int64_t indent) {
    cache.free(operand);
    return "";
}

std::string AllocateCuda::code(Cache &cache, std::string &code,
                               int64_t indent) {
    Block *block = cache.allocate(operand);
    std::string cachePosStr =
        "((" + dataTypeStr(dataType) + " *)(" + cache.cacheName + " + " +
        std::to_string(block->blockStart) + "))[" + platform.threadId() + "]";
    return cachePosStr;
}

/**
 * Register Micros
 */
REGISTER_MICRO(OperatorType::LOAD, Platform::CUDA, LoadCuda::makeObj)
REGISTER_MICRO(OperatorType::ALLOCATE, Platform::CUDA, AllocateCuda::makeObj)
REGISTER_MICRO(OperatorType::STORE, Platform::CUDA, StoreCuda::makeObj)
REGISTER_MICRO(OperatorType::FREE, Platform::CUDA, FreeCuda::makeObj)

} // namespace infini
