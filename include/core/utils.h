#pragma once
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <atomic>
#include "core/log.h"
#include "core/type.h"
#include "core/cache.h"
#include "core/platform.h"

#ifndef TOKENPASTE
#define _TOKENPASTE(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) _TOKENPASTE(x, y, z)
#endif

#ifndef PAD_UP
#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))
#endif

#ifndef PAD_DOWN
#define PAD_DOWN(x, y) (((x) / (y)) * (y))
#endif

#ifndef DIV_UP
#define DIV_UP(x, y) ((x) % (y) > 0 ? ((x) / (y) + 1) : ((x) / (y)))
#endif

#ifndef DIV_DOWN
#define DIV_DOWN(x, y) ((x) / (y))
#endif

#ifndef CEIL_ALIGN
#define CEIL_ALIGN(x, align) (((x) + (align)-1) / (align) * (align))
#endif

#ifndef FLOOR_ALIGN
#define FLOOR_ALIGN(x, align) ((x) / (align) * (align))
#endif

#ifndef ASSERT
#define ASSERT(x) assert(x)
#endif

#ifndef LOG
#define LOG(severity) PRINTLOG(Codegen, severity)
#endif

#ifndef LOG_N
#define LOG_N(severity, n)                                            \
  static std::atomic<int> TOKENPASTE(LOG_, __LINE__, _OCCURRENCE)(0); \
  if (TOKENPASTE(LOG_, __LINE__, _OCCURRENCE)++ < n) PRINTLOG(Codegen, severity)
#endif

#ifndef DLOG
#define DLOG(level) DEVELOPLOG(Codegen, level)
#endif

std::ofstream& LOG_FILE(std::string file_path);

#ifndef VECTOR_COMPARE
#define VECTOR_COMPARE(OP)                                     \
  template <class T>                                           \
  std::vector<bool> operator OP(const std::vector<T>& left,    \
                                const std::vector<T>& right) { \
    ASSERT((left).size() == (right).size());                   \
    std::vector<bool> result;                                  \
    result.reserve((left).size());                             \
    for (size_t i = 0; i < (left).size(); i++) {               \
      result.push_back(left[i] OP right[i]);                   \
    }                                                          \
    return std::move(result);                                  \
  }
#endif

#ifndef VECTOR_COMPUTE
#define VECTOR_COMPUTE(OP)                                  \
  template <class T>                                        \
  std::vector<T> operator OP(const std::vector<T>& left,    \
                             const std::vector<T>& right) { \
    ASSERT((left).size() == (right).size());                \
    std::vector<T> result;                                  \
    result.reserve((left).size());                          \
    for (size_t i = 0; i < (left).size(); i++) {            \
      result.push_back(left[i] OP right[i]);                \
    }                                                       \
    return std::move(result);                               \
  }
#endif

#ifndef VECTOR_INPLACE_COMPUTE
#define VECTOR_INPLACE_COMPUTE(OP)                           \
  template <class T>                                         \
  std::vector<T>& operator OP(std::vector<T>& left,          \
                              const std::vector<T>& right) { \
    ASSERT((left).size() == (right).size());                 \
    for (size_t i = 0; i < (left).size(); i++) {             \
      left[i] OP right[i];                                   \
    }                                                        \
    return left;                                             \
  }
#endif

#ifndef VECTOR_INPLACE_COMPUTE_SCALAR
#define VECTOR_INPLACE_COMPUTE_SCALAR(OP)                             \
  template <class T>                                                  \
  std::vector<T>& operator OP(std::vector<T>& left, const T& right) { \
    for (size_t i = 0; i < (left).size(); i++) {                      \
      left[i] OP right;                                               \
    }                                                                 \
    return left;                                                      \
  }
#endif

#ifndef CHECK
#define CHECK(condition, ...)                                     \
  if (!(condition)) {                                             \
    LOG(ERROR) << " Check failed: " #condition ". " #__VA_ARGS__; \
  }
#define CHECK_EQ(val1, val2, ...)                                         \
  if (!(val1 == val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " == " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_NE(val1, val2, ...)                                         \
  if (!(val1 != val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " != " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_LE(val1, val2, ...)                                         \
  if (!(val1 <= val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " <= " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_LT(val1, val2, ...)                                        \
  if (!(val1 < val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " < " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_GE(val1, val2, ...)                                         \
  if (!(val1 >= val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " >= " #val2 ". " #__VA_ARGS__; \
  }
#define CHECK_GT(val1, val2, ...)                                        \
  if (!(val1 > val2)) {                                                  \
    LOG(ERROR) << " Check failed: " #val1 " > " #val2 ". " #__VA_ARGS__; \
  }
#endif

namespace infini {

VECTOR_COMPARE(<)
VECTOR_COMPARE(>)
VECTOR_COMPARE(==)
VECTOR_COMPARE(<=)
VECTOR_COMPARE(>=)
VECTOR_COMPARE(!=)

VECTOR_COMPUTE(+)
VECTOR_COMPUTE(-)
VECTOR_COMPUTE(*)
VECTOR_COMPUTE(/)
VECTOR_COMPUTE(%)

VECTOR_INPLACE_COMPUTE(+=)
VECTOR_INPLACE_COMPUTE(-=)
VECTOR_INPLACE_COMPUTE(*=)
VECTOR_INPLACE_COMPUTE(/=)
VECTOR_INPLACE_COMPUTE(%=)

VECTOR_INPLACE_COMPUTE_SCALAR(+=)
VECTOR_INPLACE_COMPUTE_SCALAR(-=)
VECTOR_INPLACE_COMPUTE_SCALAR(*=)
VECTOR_INPLACE_COMPUTE_SCALAR(/=)
VECTOR_INPLACE_COMPUTE_SCALAR(%=)

bool ANY(const std::vector<bool>& boolvec);
bool ALL(const std::vector<bool>& boolvec);

std::vector<std::string> STRING_SPLIT(const std::string& input, char delimiter);

bool operator<(const Cacheline& left, const Cacheline& right);

bool operator>(const Cacheline& left, const Cacheline& right);

bool operator==(const Cacheline& left, const Cacheline& right);

int64_t VECTOR_SUM(const std::vector<int64_t>& left);

int64_t VECTOR_PRODUCT(const std::vector<int64_t>& left);

int64_t DOT_PRODUCT(const std::vector<int64_t>& left,
                    const std::vector<int64_t>& right);

std::string operator*(const std::string& left, const int64_t& right);

std::string TO_STRING(MemoryDispatch dispatch);

std::string TO_STRING(TensorDatatype datatype);

std::string TO_STRING(TensorLayout layout);

std::string TO_STRING(TensorType type);

std::string TO_STRING(const std::vector<int64_t>& input);

std::string TO_STRING(OperatorType type);

std::string TO_STRING(KernelType type);

std::string TO_STRING(Block block);

std::string TO_STRING(CacheData data);

std::string TO_STRING(CacheType type);

std::string TO_STRING(CacheHitLocation location);

std::string TO_STRING(PlatformType type);

std::string TO_STRING(Platform p);

std::string datatype_string(TensorDatatype type);

std::string size_in_bytes(int64_t size, TensorDatatype type);

std::string indentation(int64_t num);

std::string left_pad(std::string s, size_t len, char c);

std::string right_pad(std::string s, size_t len, char c);

std::string left_right_pad(std::string s, size_t len, char c);

std::string string_gather(std::vector<std::string>& strings,
                          const std::string& delimiter = ", ");

bool getBoolEnvironmentVariable(const std::string& str, bool default_value);

int64_t getLevelEnvironmentVariable(const std::string& str,
                                    int64_t default_value);

}  // namespace infini
