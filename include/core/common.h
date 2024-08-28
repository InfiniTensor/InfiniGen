#ifndef COMMON_H
#define COMMON_H
#include <cstdint>
#include <string>
#include <vector>

namespace infini {

using Shape = std::vector<int64_t>;

enum class TensorDataType { CHAR, HALF, FLOAT, DOUBLE, UNKNOWN };

enum class MicroType { BINARY, UNARY, REDUCE, BROADCAST, MEMORY };

enum class CachePolicy { LRU, LFU, FIFO, DEFAULT };

enum class OperatorType {
    // Binary
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    AND,
    OR,
    XOR,
    FLOORMOD,
    FLOORDIV,
    // Unary
    SIGMOID,
    RELU,
    SQRT,
    RSQRT,
    RECIP,
    SIN,
    COS,
    TANH,
    // Memory
    LOAD,
    ALLOCATE,
    STORE,
    FREE,
    // Sync
    SYNC,
    // Default
    UNKNOWN
};

} // namespace infini

#endif
