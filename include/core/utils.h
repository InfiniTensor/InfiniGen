#ifndef UTILS_H
#define UTILS_H
#include "core/common.h"
#include "core/log.h"
#include "core/platform.h"
#include <cassert>
#include <vector>

#ifndef TOKENPASTE
#define _TOKENPASTE(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) _TOKENPASTE(x, y, z)
#endif

#ifndef TOKENCAT
#define _CAT(A, B) A##B
#define CAT(A, B) _CAT(A, B)
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
#define LOG(severity) PRINTLOG(InfiniGen, severity)
#endif

#ifndef LOG_N
#define LOG_N(severity, n)                                                     \
    static std::atomic<int> TOKENPASTE(LOG_, __LINE__, _OCCURRENCE)(0);        \
    if (TOKENPASTE(LOG_, __LINE__, _OCCURRENCE)++ < n)                         \
    PRINTLOG(InfiniGen, severity)
#endif

#ifndef DLOG
#define DLOG(level) DEVELOPLOG(InfiniGen, level)
#endif

#ifndef VECTOR_COMPARE
#define VECTOR_COMPARE(OP)                                                     \
    template <class T>                                                         \
    std::vector<bool> operator OP(const std::vector<T> &left,                  \
                                  const std::vector<T> &right) {               \
        ASSERT((left).size() == (right).size());                               \
        std::vector<bool> result;                                              \
        result.reserve((left).size());                                         \
        for (size_t i = 0; i < (left).size(); i++) {                           \
            result.push_back(left[i] OP right[i]);                             \
        }                                                                      \
        return std::move(result);                                              \
    }
#endif

#ifndef VECTOR_COMPUTE
#define VECTOR_COMPUTE(OP)                                                     \
    template <class T>                                                         \
    std::vector<T> operator OP(const std::vector<T> &left,                     \
                               const std::vector<T> &right) {                  \
        ASSERT((left).size() == (right).size());                               \
        std::vector<T> result;                                                 \
        result.reserve((left).size());                                         \
        for (size_t i = 0; i < (left).size(); i++) {                           \
            result.push_back(left[i] OP right[i]);                             \
        }                                                                      \
        return std::move(result);                                              \
    }
#endif

#ifndef VECTOR_COMPUTE_SCALAR
#define VECTOR_COMPUTE_SCALAR(OP)                                              \
    template <class T>                                                         \
    std::vector<T> operator OP(const std::vector<T> &left, const T &right) {   \
        std::vector<T> result;                                                 \
        result.reserve((left).size());                                         \
        for (size_t i = 0; i < (left).size(); i++) {                           \
            result.push_back(left[i] OP right);                                \
        }                                                                      \
        return std::move(result);                                              \
    }
#endif

#ifndef VECTOR_INPLACE_COMPUTE
#define VECTOR_INPLACE_COMPUTE(OP)                                             \
    template <class T>                                                         \
    std::vector<T> &operator OP(std::vector<T> &left,                          \
                                const std::vector<T> &right) {                 \
        ASSERT((left).size() == (right).size());                               \
        for (size_t i = 0; i < (left).size(); i++) {                           \
            left[i] OP right[i];                                               \
        }                                                                      \
        return left;                                                           \
    }
#endif

#ifndef VECTOR_INPLACE_COMPUTE_SCALAR
#define VECTOR_INPLACE_COMPUTE_SCALAR(OP)                                      \
    template <class T>                                                         \
    std::vector<T> &operator OP(std::vector<T> &left, const T &right) {        \
        for (size_t i = 0; i < (left).size(); i++) {                           \
            left[i] OP right;                                                  \
        }                                                                      \
        return left;                                                           \
    }
#endif

#ifndef CHECK
#define CHECK(condition, ...)                                                  \
    if (!(condition)) {                                                        \
        LOG(ERROR) << " Check failed: " #condition ". " #__VA_ARGS__;          \
    }
#define CHECK_EQ(val1, val2, ...)                                              \
    if (!(val1 == val2)) {                                                     \
        LOG(ERROR) << " Check failed: " #val1 " == " #val2 ". " #__VA_ARGS__;  \
    }
#define CHECK_NE(val1, val2, ...)                                              \
    if (!(val1 != val2)) {                                                     \
        LOG(ERROR) << " Check failed: " #val1 " != " #val2 ". " #__VA_ARGS__;  \
    }
#define CHECK_LE(val1, val2, ...)                                              \
    if (!(val1 <= val2)) {                                                     \
        LOG(ERROR) << " Check failed: " #val1 " <= " #val2 ". " #__VA_ARGS__;  \
    }
#define CHECK_LT(val1, val2, ...)                                              \
    if (!(val1 < val2)) {                                                      \
        LOG(ERROR) << " Check failed: " #val1 " < " #val2 ". " #__VA_ARGS__;   \
    }
#define CHECK_GE(val1, val2, ...)                                              \
    if (!(val1 >= val2)) {                                                     \
        LOG(ERROR) << " Check failed: " #val1 " >= " #val2 ". " #__VA_ARGS__;  \
    }
#define CHECK_GT(val1, val2, ...)                                              \
    if (!(val1 > val2)) {                                                      \
        LOG(ERROR) << " Check failed: " #val1 " > " #val2 ". " #__VA_ARGS__;   \
    }
#endif

std::ofstream &LOG_FILE(std::string file_path);

namespace infini {

void COMPILE(std::string input_file_path, std::string output_binary_directory,
             Platform platform);

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

VECTOR_COMPUTE_SCALAR(+)
VECTOR_COMPUTE_SCALAR(-)
VECTOR_COMPUTE_SCALAR(*)
VECTOR_COMPUTE_SCALAR(/)
VECTOR_COMPUTE_SCALAR(%)

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

bool ANY_TRUE(const std::vector<bool> &input);

bool ALL_TRUE(const std::vector<bool> &input);

int64_t VECTOR_SUM(const std::vector<int64_t> &left);

int64_t VECTOR_PRODUCT(const std::vector<int64_t> &left);

int64_t VECTOR_DOT_PRODUCT(const std::vector<int64_t> &left,
                           const std::vector<int64_t> &right);

std::vector<int64_t> MINIMUM(const std::vector<int64_t> &left,
                             const std::vector<int64_t> &right);

std::vector<int64_t> MAXIMUM(const std::vector<int64_t> &left,
                             const std::vector<int64_t> &right);

std::string TO_STRING(TensorDataType datatype);

std::string dataTypeStr(TensorDataType datatype);

std::string TO_STRING(OperatorType operatortype);

std::string TO_STRING(MicroType microtype);

std::string TO_STRING(CachePolicy cachePolicy);

std::string INITIALIZER(const std::vector<int64_t> &input);

std::string TO_STRING(const std::vector<int64_t> &input);

std::string TO_STRING(const std::vector<std::string> &input);

std::string TO_STRING(const bool input);

std::string operator*(const std::string &left, const int64_t &right);

int64_t SIZE_OF(TensorDataType datatype);

std::vector<int64_t> CALCULATE_STRIDE(const std::vector<int64_t> &shape);

std::string INDENTATION(int64_t num);

std::string STRING_GATHER(std::vector<std::string> &strings,
                          const std::string &delimiter = ", ");

std::vector<std::string> STRING_SPLIT(const std::string &input, char delimiter);

} // namespace infini

#endif
