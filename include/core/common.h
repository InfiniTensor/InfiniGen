#ifndef COMMON_H
#define COMMON_H
#include <cstdint>
#include <string>
#include <vector>

namespace infini {

using Shape = std::vector<int64_t>;

enum class TensorDataType { CHAR, HALF, FLOAT, DOUBLE, UNKNOWN };

} // namespace infini

#endif