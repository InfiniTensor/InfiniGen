#ifndef COMMON_H
#define COMMON_H
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace infini {

using Shape = std::vector<int64_t>;

using Attribute =
    std::variant<int32_t, int64_t, bool, float, double, void *, std::string,
                 std::vector<int32_t>, std::vector<int64_t>, std::vector<bool>,
                 std::vector<float>, std::vector<double>, std::vector<void *>,
                 std::vector<std::string>>;

enum class TensorDataType { CHAR, HALF, FLOAT, DOUBLE, UNKNOWN };

enum class MicroType {
    BINARY,
    UNARY,
    REDUCE,
    BROADCAST,
    BROADCAST_BINARY,
    MEMORY,
    SYNC
};

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
    // Broadcast
    BROADCAST,
    BROADCAST_ADD,
    BROADCAST_SUB,
    BROADCAST_MUL,
    BROADCAST_DIV,
    // Reduce
    REDUCE,
    // Sync
    SYNC,
    // Default
    UNKNOWN
};

} // namespace infini

#endif
