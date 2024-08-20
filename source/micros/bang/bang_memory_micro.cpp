#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadBang::code(Cache &cache, std::string &code, int64_t indent) {
    std::string lengthStr = std::to_string(lengthInBytes);

    bool cached = (cache.find(operand) != nullptr);
    Block *block = cache.load(operand);
    std::string cachePosStr = "(" + dataTypeStr(dataType) + " *)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + ")";
    if (cached) {
        return cachePosStr;
    }

    std::string TileOffsetStr = operand->tensor->tensorName + " + " +
                                std::to_string(operand->tileOffset);

    if (operand->tileShape.size() == 1) {
        code += INDENTATION(indent) + "__memcpy(" + cachePosStr + ", " +
                TileOffsetStr + ", " + lengthStr + ", GDRAM2NRAM);\n";
    } else if (operand->tileShape.size() == 2) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[0] * SIZE_OF(dataType));
        code += INDENTATION(indent) + "__memcpy(" + cachePosStr + ", " +
                TileOffsetStr + ", " + segLengthStr + ", GDRAM2NRAM, " +
                std::to_string(operand->tileStride[0]) + ", " +
                std::to_string(operand->tensor->tensorStride[0]) + ", " +
                std::to_string(operand->tileShape[0] - 1) + ");\n";
    } else if (operand->tileShape.size() == 3) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[1] * SIZE_OF(dataType));
        code += INDENTATION(indent) + "__memcpy(" + cachePosStr + ", " +
                TileOffsetStr + ", " + segLengthStr + ", GDRAM2NRAM, " +
                std::to_string(operand->tileStride[1]) + ", " +
                std::to_string(operand->tileShape[1] - 1) + ", " +
                std::to_string(operand->tileStride[0]) + ", " +
                std::to_string(operand->tileShape[0] - 1) + ", " +
                std::to_string(operand->tensor->tensorStride[1]) + ", " +
                std::to_string(operand->tensor->tensorShape[1] - 1) + ", " +
                std::to_string(operand->tensor->tensorStride[0]) + ", " +
                std::to_string(operand->tensor->tensorShape[0] - 1) + ");\n";
    } else {
        LOG(ERROR) << "\"__memcpy\" only supports up to 3D.";
    }
    return cachePosStr;
}

std::string StoreBang::code(Cache &cache, std::string &code, int64_t indent) {
    std::string lengthStr = std::to_string(lengthInBytes);

    Block *block = cache.load(operand);
    std::string cachePosStr = "(" + dataTypeStr(dataType) + " *)(" +
                              cache.cacheName + " + " +
                              std::to_string(block->blockStart) + ")";

    std::string TileOffsetStr = operand->tensor->tensorName + " + " +
                                std::to_string(operand->tileOffset);

    if (operand->tileShape.size() == 1) {
        code += INDENTATION(indent) + "__memcpy(" + TileOffsetStr + ", " +
                cachePosStr + ", " + lengthStr + ", NRAM2GDRAM);\n";
    } else if (operand->tileShape.size() == 2) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[0] * SIZE_OF(dataType));
        code += INDENTATION(indent) + "__memcpy(" + TileOffsetStr + ", " +
                cachePosStr + ", " + segLengthStr + ", GDRAM2NRAM, " +
                std::to_string(operand->tensor->tensorStride[0]) + ", " +
                std::to_string(operand->tileStride[0]) + ", " +
                std::to_string(operand->tileShape[0] - 1) + ");\n";
    } else if (operand->tileShape.size() == 3) {
        std::string segLengthStr =
            std::to_string(operand->tileStride[1] * SIZE_OF(dataType));
        code += INDENTATION(indent) + "__memcpy(" + TileOffsetStr + ", " +
                cachePosStr + ", " + segLengthStr + ", GDRAM2NRAM, " +
                std::to_string(operand->tensor->tensorStride[1]) + ", " +
                std::to_string(operand->tensor->tensorShape[1] - 1) + ", " +
                std::to_string(operand->tensor->tensorStride[0]) + ", " +
                std::to_string(operand->tensor->tensorShape[0] - 1) + ", " +
                std::to_string(operand->tileStride[1]) + ", " +
                std::to_string(operand->tileShape[1] - 1) + ", " +
                std::to_string(operand->tileStride[0]) + ", " +
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

/* GET OFFSET OF A TILE IN TENSOR USING TASKID() */
// std::vector<uint64_t> tileCoords;
// std::vector<uint64_t> tileGridStride = operand->tileGridStride;
// uint64_t index = platform.taskId();
// for (uint64_t i = 0; i < tileGridStride.size(); i++) {
//     tileCoords.push_back((uint64_t)(index / tileGridStride[i]));
//     index %= tileGridStride[i];
// }
// std::vector<uint64_t> tileShape = operand->tileShape;
// std::vector<uint64_t> tileStride = operand->tileStride;
// uint64_t offset = 0;
// for (uint64_t i = 0; i < tileCoords.size(); i++) {
//     offset += tileCoords[i] * tileShape[i] * tileStride[i];
// }
