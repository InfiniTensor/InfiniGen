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
  ADD,
  SUB,
  MUL,
  SIGMOID,
  RELU,
  LOAD,
  ALLOCATE,
  STORE,
  FREE,
  SYNC,
  SQRT,
  RSQRT
};
// KernelType
enum class KernelType { BINARY, UNARY, REDUCE, BROADCAST, MEMORY, FMA, SYNC };

// CacheType
enum class CacheType { CACHE, LDRAM };

// CacheHitLocation
enum class CacheHitLocation { CACHE, LDRAM, NOT_FOUND, ERROR };
}  // namespace infini
