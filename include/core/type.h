#pragma once
#include <tuple>
#include <string>

namespace infini {
// Cacheline
using Cacheline = std::tuple<std::string, std::string, int>;
// MemoryDispatch
enum class MemoryDispatch { RANDOM, FIFO, LRU, LFU };
enum class TensorDatatype { HALF, FLOAT, DOUBLE, INT32, TILE };
enum class TensorLayout { NCHW, NHWC, ARRAY };
enum class TensorType { CONST, VARIABLE };
// OperatorType
enum class OperatorType { ADD, SUB, MUL, SIGMOID, RELU };
// KernelType
enum class KernelType {
  BINARY,
  UNARY,
  REDUCE,
  BROADCAST,
  MEMORY,
  FMA,
  LOAD,
  CACHE,
  ADD,
  SUB,
  MUL,
  SIN,
  COS
};
// PlatformType
enum class PlatformType { CUDA, BANG };
}  // namespace infini
