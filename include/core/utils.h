#pragma once
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <atomic>
#include "core/log.h"
#include "core/type.h"

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
#define LOG(severity) PRINTLOG(Thread, severity)
#endif

#ifndef LOG_N
#define LOG_N(severity, n)                                            \
  static std::atomic<int> TOKENPASTE(LOG_, __LINE__, _OCCURRENCE)(0); \
  if (TOKENPASTE(LOG_, __LINE__, _OCCURRENCE)++ < n) PRINTLOG(Thread, severity)
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

#ifndef DOT_PRODUCT
#define DOT_PRODUCT(vec1, vec2)                  \
  ({                                             \
    ASSERT((vec1).size() == (vec2).size());      \
    int64_t result = 0;                          \
    for (size_t i = 0; i < (vec1).size(); ++i) { \
      result += (vec1)[i] * (vec2)[i];           \
    }                                            \
    result;                                      \
  })
#endif

namespace infini {

std::vector<std::string> STRING_SPLIT(const std::string& input, char delimiter);

bool operator<(const Cacheline& left, const Cacheline& right);

bool operator>(const Cacheline& left, const Cacheline& right);

bool operator==(const Cacheline& left, const Cacheline& right);

int64_t VECTOR_SUM(const std::vector<int64_t>& left);

int64_t VECTOR_PRODUCT(const std::vector<int64_t>& left);

bool ALL_EQLESS(const std::vector<int64_t>& left,
                const std::vector<int64_t>& right);

std::vector<int64_t> operator+(const std::vector<int64_t>& left,
                               const std::vector<int64_t>& right);
std::vector<int64_t> operator-(const std::vector<int64_t>& left,
                               const std::vector<int64_t>& right);
std::vector<int64_t> operator*(const std::vector<int64_t>& left,
                               const std::vector<int64_t>& right);
std::vector<int64_t> operator/(const std::vector<int64_t>& left,
                               const std::vector<int64_t>& right);
std::vector<int64_t> operator%(const std::vector<int64_t>& left,
                               const std::vector<int64_t>& right);
std::vector<int64_t>& operator+=(std::vector<int64_t>& left,
                                 const int64_t& right);
std::vector<int64_t>& operator-=(std::vector<int64_t>& left,
                                 const int64_t& right);
std::vector<int64_t>& operator*=(std::vector<int64_t>& left,
                                 const int64_t& right);
std::vector<int64_t>& operator/=(std::vector<int64_t>& left,
                                 const int64_t& right);
std::vector<int64_t>& operator%=(std::vector<int64_t>& left,
                                 const int64_t& right);
std::vector<int64_t>& operator+=(std::vector<int64_t>& left,
                                 const std::vector<int64_t>& right);
std::vector<int64_t>& operator-=(std::vector<int64_t>& left,
                                 const std::vector<int64_t>& right);
std::vector<int64_t>& operator*=(std::vector<int64_t>& left,
                                 const std::vector<int64_t>& right);
std::vector<int64_t>& operator/=(std::vector<int64_t>& left,
                                 const std::vector<int64_t>& right);
std::vector<int64_t>& operator%=(std::vector<int64_t>& left,
                                 const std::vector<int64_t>& right);

std::string operator*(const std::string& left, const int64_t& right);

std::string TO_STRING(MemoryDispatch dispatch);

std::string TO_STRING(TensorDatatype datatype);

std::string TO_STRING(TensorLayout layout);

std::string TO_STRING(TensorType type);

std::string TO_STRING(std::vector<int64_t>& input);

std::string TO_STRING(OperatorType type);

std::string TO_STRING(KernelType type);

std::string TO_STRING(PlatformType type);

std::string datatype_string(TensorDatatype type);

std::string indentation(int64_t num);

}  // namespace infini
