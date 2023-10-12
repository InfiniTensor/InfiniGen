#pragma once
#include <tuple>
#include <string>

namespace infini {
// Cacheline
using Cacheline = std::tuple<std::string, std::string, int>;
// MemoryDispatch
enum class MemoryDispatch { RANDOM, FIFO, LRU, LFU };
enum class TensorDatatype { HALF, FLOAT, DOUBLE, INT32 };
enum class TensorLayout { NCHW, NHWC, ARRAY };
enum class TensorType { CONST, VARIABLE };
// OperatorType
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
  // Unary
  SIGMOID,
  RELU,
  // Memory
  LOAD,
  ALLOCATE,
  STORE,
  FREE,
  // Sync
  SYNC
};
// KernelType
enum class KernelType { BINARY, UNARY, REDUCE, BROADCAST, MEMORY, FMA, SYNC };

// CacheType
enum class CacheType { CACHE, LDRAM };

// CacheHitLocation
enum class CacheHitLocation { CACHE, LDRAM, NOT_FOUND, ERROR };
}  // namespace infini
