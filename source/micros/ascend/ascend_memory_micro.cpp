#include "core/cache.h"
#include "core/tile.h"
#include "core/utils.h"
#include "micros/memory_micro.h"

namespace infini {

std::string LoadAscend::code(Cache &cache, std::string &code, int64_t indent) {
    int64_t dataTypeSize = SIZE_OF(dataType);
    std::string lengthStr = std::to_string(length);

    bool cached = (cache.find(operand) != nullptr);
    Block *block = cache.load(operand);
    std::string cachePosStr =
        cache.cacheName + "[" + std::to_string(block->blockStart) + "]";
    if (cached) {
        return cachePosStr;
    }

    std::string tensorName = operand->tensor->tensorName;

    std::string gmDecl =
        platform.glmemDecl(dataTypeStr(dataType), tensorName + "Gm;\n");

    std::string gmInit = tensorName + "Gm.SetGlobalBuffer(" + tensorName +
                         " + " +
                         operand->getOffsetInTensor(operand->tileCoordsExpr) +
                         ", " + lengthStr + ");\n";

    code += INDENTATION(indent) + gmDecl;
    code += INDENTATION(indent) + gmInit;

    std::string tileOffsetStr =
        operand->tensor->tensorName + "Gm[" +
        operand->getOffsetInTensor(operand->tileCoordsExpr) + "]";

    if (operand->tileShape.size() == 1) {
        // TODO: 这里没有考虑 length * dataTypeSize 不是32字节整的情况
        code += INDENTATION(indent) + "DataCopy(" + cachePosStr + ", " +
                tileOffsetStr + ", " + lengthStr + ");\n";
    } else {
        LOG(ERROR) << "\"DataCopy\" only supports 1D.";
    }
    return cachePosStr;
}

std::string StoreAscend::code(Cache &cache, std::string &code, int64_t indent) {
    std::string lengthStr = std::to_string(length);

    Block *block = cache.load(operand);
    std::string cachePosStr =
        cache.cacheName + "[" + std::to_string(block->blockStart) + "]";
    std::string tensorName = operand->tensor->tensorName;
    std::string gmDecl =
        platform.glmemDecl(dataTypeStr(dataType), tensorName + "Gm;\n");

    std::string gmInit = tensorName + "Gm.SetGlobalBuffer(" + tensorName +
                         " + " +
                         operand->getOffsetInTensor(operand->tileCoordsExpr) +
                         ", " + lengthStr + ");\n";

    code += INDENTATION(indent) + gmDecl;
    code += INDENTATION(indent) + gmInit;

    std::string tileOffsetStr =
        operand->tensor->tensorName + "Gm[" +
        operand->getOffsetInTensor(operand->tileCoordsExpr) + "]";

    if (operand->tileShape.size() == 1) {
        // TODO: 这里没有考虑 length * dataTypeSize 不是32字节整的情况
        code += INDENTATION(indent) + "DataCopy(" + tileOffsetStr + ", " +
                cachePosStr + ", " + lengthStr + ");\n";
    } else {
        LOG(ERROR) << "\"DataCopy\" only supports 1D.";
    }
    return cachePosStr;
}

std::string FreeAscend::code(Cache &cache, std::string &code, int64_t indent) {
    cache.free(operand);
    return "";
}

std::string AllocateAscend::code(Cache &cache, std::string &code,
                                 int64_t indent) {
    Block *block = cache.allocate(operand);
    std::string cachePosStr =
        cache.cacheName + "[" + std::to_string(block->blockStart) + "]";
    return cachePosStr;
}

REGISTER_MICRO(OperatorType::LOAD, Platform::ASCEND, LoadAscend::makeObj)
REGISTER_MICRO(OperatorType::ALLOCATE, Platform::ASCEND,
               AllocateAscend::makeObj)
REGISTER_MICRO(OperatorType::STORE, Platform::ASCEND, StoreAscend::makeObj)
REGISTER_MICRO(OperatorType::FREE, Platform::ASCEND, FreeAscend::makeObj)

} // namespace infini
