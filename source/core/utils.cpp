#include "core/utils.h"

namespace infini {

std::vector<std::string> STRING_SPLIT(const std::string &input,
                                      char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream data(input);
  std::string token;
  while (std::getline(data, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

bool operator<(const Cacheline &left, const Cacheline &right) {
  return std::get<2>(left) < std::get<2>(right);
}

bool operator>(const Cacheline &left, const Cacheline &right) {
  return std::get<2>(left) > std::get<2>(right);
}

bool operator==(const Cacheline &left, const Cacheline &right) {
  return std::get<2>(left) == std::get<2>(right);
}

int64_t VECTOR_SUM(const std::vector<int64_t> &left) {
  int64_t result = 0;
  for (auto i = 0; i < left.size(); ++i) {
    result += left[i];
  }
  return result;
}

int64_t VECTOR_PRODUCT(const std::vector<int64_t> &left) {
  int64_t result = 1;
  for (auto i = 0; i < left.size(); ++i) {
    result *= left[i];
  }
  return result;
}

std::vector<int64_t> operator+(const std::vector<int64_t> &left,
                               const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  std::vector<int64_t> result(left.size(), 0);
  for (auto i = 0; i < left.size(); ++i) {
    result[i] = left[i] + right[i];
  }
  return std::move(result);
}

std::vector<int64_t> operator-(const std::vector<int64_t> &left,
                               const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  std::vector<int64_t> result(left.size(), 0);
  for (auto i = 0; i < left.size(); ++i) {
    result[i] = left[i] - right[i];
  }
  return std::move(result);
}

std::vector<int64_t> operator*(const std::vector<int64_t> &left,
                               const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  std::vector<int64_t> result(left.size(), 0);
  for (auto i = 0; i < left.size(); ++i) {
    result[i] = left[i] * right[i];
  }
  return std::move(result);
}

std::vector<int64_t> operator/(const std::vector<int64_t> &left,
                               const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  std::vector<int64_t> result(left.size(), 0);
  for (auto i = 0; i < left.size(); ++i) {
    result[i] = left[i] / right[i];
  }
  return std::move(result);
}

std::vector<int64_t> operator%(const std::vector<int64_t> &left,
                               const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  std::vector<int64_t> result(left.size(), 0);
  for (auto i = 0; i < left.size(); ++i) {
    result[i] = left[i] % right[i];
  }
  return std::move(result);
}

std::vector<int64_t> &operator+=(std::vector<int64_t> &left,
                                 const int64_t &right) {
  for (auto i = 0; i < left.size(); ++i) {
    left[i] += right;
  }
  return left;
}

std::vector<int64_t> &operator-=(std::vector<int64_t> &left,
                                 const int64_t &right) {
  for (auto i = 0; i < left.size(); ++i) {
    left[i] -= right;
  }
  return left;
}

std::vector<int64_t> &operator*=(std::vector<int64_t> &left,
                                 const int64_t &right) {
  for (auto i = 0; i < left.size(); ++i) {
    left[i] *= right;
  }
  return left;
}

std::vector<int64_t> &operator/=(std::vector<int64_t> &left,
                                 const int64_t &right) {
  for (auto i = 0; i < left.size(); ++i) {
    left[i] /= right;
  }
  return left;
}

std::vector<int64_t> &operator%=(std::vector<int64_t> &left,
                                 const int64_t &right) {
  for (auto i = 0; i < left.size(); ++i) {
    left[i] %= right;
  }
  return left;
}

std::vector<int64_t> &operator+=(std::vector<int64_t> &left,
                                 const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  for (auto i = 0; i < left.size(); ++i) {
    left[i] = left[i] + right[i];
  }
  return left;
}

std::vector<int64_t> &operator-=(std::vector<int64_t> &left,
                                 const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  for (auto i = 0; i < left.size(); ++i) {
    left[i] = left[i] - right[i];
  }
  return left;
}

std::vector<int64_t> &operator*=(std::vector<int64_t> &left,
                                 const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  for (auto i = 0; i < left.size(); ++i) {
    left[i] = left[i] * right[i];
  }
  return left;
}

std::vector<int64_t> &operator/=(std::vector<int64_t> &left,
                                 const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  for (auto i = 0; i < left.size(); ++i) {
    left[i] = left[i] / right[i];
  }
  return left;
}

std::vector<int64_t> &operator%=(std::vector<int64_t> &left,
                                 const std::vector<int64_t> &right) {
  CHECK_EQ(left.size(), right.size());
  for (auto i = 0; i < left.size(); ++i) {
    left[i] = left[i] % right[i];
  }
  return left;
}

std::string operator*(const std::string &left, const int64_t &right) {
  std::string result = "";
  for (auto i = 0; i < right; ++i) {
    result += left;
  }
  return std::move(result);
}

std::string TO_STRING(MemoryDispatch dispatch) {
  switch (dispatch) {
    case MemoryDispatch::RANDOM:
      return "RANDOM";
    case MemoryDispatch::FIFO:
      return "FIFO";
    case MemoryDispatch::LRU:
      return "LRU";
    case MemoryDispatch::LFU:
      return "LFU";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(TensorDatatype datatype) {
  switch (datatype) {
    case TensorDatatype::HALF:
      return "HALF";
    case TensorDatatype::FLOAT:
      return "FLOAT";
    case TensorDatatype::DOUBLE:
      return "DOUBLE";
    case TensorDatatype::INT32:
      return "INT32";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::NCHW:
      return "NCHW";
    case TensorLayout::NHWC:
      return "NHWC";
    case TensorLayout::ARRAY:
      return "ARRAY";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(TensorType type) {
  switch (type) {
    case TensorType::CONST:
      return "CONST";
    case TensorType::VARIABLE:
      return "VARIABLE";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(std::vector<int64_t> &input) {
  std::string info_string = "[";
  for (auto i = 0; i < input.size(); ++i) {
    info_string += std::to_string(input[i]);
    info_string += (i == (input.size() - 1) ? "" : ",");
  }
  info_string += "]";
  return info_string;
}

std::string TO_STRING(OperatorType type) {
#define CASE(NAME)         \
  case OperatorType::NAME: \
    return #NAME
  switch (type) {
    CASE(ADD);
    CASE(SUB);
    CASE(MUL);
    CASE(SIGMOID);
    CASE(RELU);
    default:
      return "UNKNOWN";
  }
#undef CASE
}

std::string TO_STRING(KernelType type) {
#define CASE(NAME)       \
  case KernelType::NAME: \
    return #NAME
  switch (type) {
    CASE(BINARY);
    CASE(UNARY);
    CASE(REDUCE);
    CASE(BROADCAST);
    CASE(MEMORY);
    CASE(FMA);
    default:
      return "UNKNOWN";
  }
#undef CASE
}

std::string TO_STRING(PlatformType type) {
  switch (type) {
    case PlatformType::CUDA:
      return "CUDA";
    case PlatformType::BANG:
      return "BANG";
    default:
      return "UNKNOWN";
  }
}

std::string datatype_string(TensorDatatype datatype) {
  switch (datatype) {
    case TensorDatatype::HALF:
      return "half";
    case TensorDatatype::FLOAT:
      return "float";
    case TensorDatatype::DOUBLE:
      return "double";
    case TensorDatatype::INT32:
      return "int";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(CacheType type) {
  switch (type) {
    case CacheType::CACHE:
      return "CACHE";
    case CacheType::LDRAM:
      return "LDRAM";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(CacheHitLocation location) {
  switch (location) {
    case CacheHitLocation::CACHE:
      return "CACHE";
    case CacheHitLocation::LDRAM:
      return "LDRAM";
    case CacheHitLocation::NOT_FOUND:
      return "NOT_FOUND";
    case CacheHitLocation::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

std::string TO_STRING(Block block) {
  std::string info = block.cache_name + "_" + TO_STRING(block.cache_type) +
                     "_" + std::to_string(block.block_offset) + "_" +
                     std::to_string(block.block_size);
  return info;
}

std::string TO_STRING(CacheData data) {
  std::string data_info = data.name + "_" + std::to_string(data.offset) + "_" +
                          std::to_string(data.size);
  return data_info;
}

std::string indentation(int64_t num) { return std::string(num * 2, ' '); }

std::string left_pad(std::string s, size_t len, char c) {
  if (s.length() >= len) {
    return s;
  }
  std::string res = std::string(len - s.length(), c) + s;
  return res;
}

std::string right_pad(std::string s, size_t len, char c) {
  if (s.length() >= len) {
    return s;
  }
  std::string res = s + std::string(len - s.length(), c);
  return res;
}

std::string left_right_pad(std::string s, size_t len, char c) {
  if (s.length() >= len) {
    return s;
  }
  std::string res = std::string((len - s.length()) / 2, c) + s;
  res += std::string(len - res.length(), c);
  return res;
}

}  // namespace infini
