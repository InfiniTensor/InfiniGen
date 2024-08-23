#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadBang::code(Cache &cache, std::string &code, int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length * dataTypeSize);

    bool cached = (cache.find(operand) != nullptr);
    Block *block = cache.load(operand);
    std::string cachePosStr = "(" + dataTypeStr(dataType) + " *)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + ")";
    if (cached) {
        return cachePosStr;
    }

    // use blockId/taskId to get offset
    std::string tileOffsetStr =
        operand->tensor->tensorName + " + " +
        platform.offset(operand->tensor->tensorStride,
                        operand->tensor->tileGridStride, operand->tileShape);

    if (operand->tileShape.size() == 1) {
        code += INDENTATION(indent) + "__memcpy(" + cachePosStr + ", " +
                tileOffsetStr + ", " + lengthStr + ", GDRAM2NRAM);\n";
    } else if (operand->tileShape.size() == 2) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[0] * SIZE_OF(dataType));
        code +=
            INDENTATION(indent) + "__memcpy(" + cachePosStr + ", " +
            tileOffsetStr + ", " + segLengthStr + ", GDRAM2NRAM, " +
            std::to_string(operand->tileStride[0] * dataTypeSize) + ", " +
            std::to_string(operand->tensor->tensorStride[0] * dataTypeSize) +
            ", " + std::to_string(operand->tileShape[0] - 1) + ");\n";
    } else if (operand->tileShape.size() == 3) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[1] * SIZE_OF(dataType));
        code +=
            INDENTATION(indent) + "__memcpy(" + cachePosStr + ", " +
            tileOffsetStr + ", " + segLengthStr + ", GDRAM2NRAM, " +
            std::to_string(operand->tileStride[1] * dataTypeSize) + ", " +
            std::to_string(operand->tileShape[1] - 1) + ", " +
            std::to_string(operand->tileStride[0] * dataTypeSize) + ", " +
            std::to_string(operand->tileShape[0] - 1) + ", " +
            std::to_string(operand->tensor->tensorStride[1] * dataTypeSize) +
            ", " + std::to_string(operand->tileShape[1] - 1) + ", " +
            std::to_string(operand->tensor->tensorStride[0] * dataTypeSize) +
            ", " + std::to_string(operand->tileShape[0] - 1) + ");\n";
    } else {
        LOG(ERROR) << "\"__memcpy\" only supports up to 3D.";
    }
    return cachePosStr;
}

std::string StoreBang::code(Cache &cache, std::string &code, int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length * dataTypeSize);

    Block *block = cache.load(operand);
    std::string cachePosStr = "(" + dataTypeStr(dataType) + " *)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + ")";

    // use blockId/taskId to get offset
    std::string tileOffsetStr =
        operand->tensor->tensorName + " + " +
        platform.offset(operand->tensor->tensorStride,
                        operand->tensor->tileGridStride, operand->tileShape);

    if (operand->tileShape.size() == 1) {
        code += INDENTATION(indent) + "__memcpy(" + tileOffsetStr + ", " +
                cachePosStr + ", " + lengthStr + ", NRAM2GDRAM);\n";
    } else if (operand->tileShape.size() == 2) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[0] * SIZE_OF(dataType));
        code +=
            INDENTATION(indent) + "__memcpy(" + tileOffsetStr + ", " +
            cachePosStr + ", " + segLengthStr + ", NRAM2GDRAM, " +
            std::to_string(operand->tensor->tensorStride[0] * dataTypeSize) +
            ", " + std::to_string(operand->tileStride[0] * dataTypeSize) +
            ", " + std::to_string(operand->tileShape[0] - 1) + ");\n";
    } else if (operand->tileShape.size() == 3) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[1] * SIZE_OF(dataType));
        code +=
            INDENTATION(indent) + "__memcpy(" + tileOffsetStr + ", " +
            cachePosStr + ", " + segLengthStr + ", NRAM2GDRAM, " +
            std::to_string(operand->tensor->tensorStride[1] * dataTypeSize) +
            ", " + std::to_string(operand->tileShape[1] - 1) + ", " +
            std::to_string(operand->tensor->tensorStride[0] * dataTypeSize) +
            ", " + std::to_string(operand->tileShape[0] - 1) + ", " +
            std::to_string(operand->tileStride[1] * dataTypeSize) + ", " +
            std::to_string(operand->tileShape[1] - 1) + ", " +
            std::to_string(operand->tileStride[0] * dataTypeSize) + ", " +
            std::to_string(operand->tileShape[0] - 1) + ");\n";
    } else {
        LOG(ERROR) << "\"__memcpy\" only supports up to 3D.";
    }
    return cachePosStr;
}

std::string FreeBang::code(Cache &cache, std::string &code, int64_t indent) {
    cache.free(operand);
    return "";
}

std::string AllocateBang::code(Cache &cache, std::string &code,
                               int64_t indent) {
    Block *block = cache.allocate(operand);
    std::string cachePosStr = "(" + dataTypeStr(dataType) + " *)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + ")";
    return cachePosStr;
}

/**
 * Register Micros
 */
REGISTER_MICRO(OperatorType::LOAD, Platform::BANG, LoadBang::makeObj)
REGISTER_MICRO(OperatorType::ALLOCATE, Platform::BANG, AllocateBang::makeObj)
REGISTER_MICRO(OperatorType::STORE, Platform::BANG, StoreBang::makeObj)
REGISTER_MICRO(OperatorType::FREE, Platform::BANG, FreeBang::makeObj)

} // namespace infini
