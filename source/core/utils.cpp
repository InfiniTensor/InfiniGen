#include "core/utils.h"
#include <algorithm>

std::ofstream &LOG_FILE(std::string file_path) {
  infini::log_stream.flush();
  infini::log_stream.close();
  infini::log_stream.open(file_path, std::ios::app | std::ios::out);
  return infini::log_stream;
}

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

int64_t DOT_PRODUCT(const std::vector<int64_t> &left,
                    const std::vector<int64_t> &right) {
  ASSERT(left.size() == right.size());
  int64_t result = 0;
  for (size_t i = 0; i < left.size(); ++i) {
    result += left[i] * right[i];
  }
  return result;
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

std::string TO_STRING(const std::vector<int64_t> &input) {
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

std::string TO_STRING(Platform p) {
  switch (p.underlying()) {
    case Platform::CUDA:
      return "CUDA";
    case Platform::BANG:
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

int64_t datatype_size(TensorDatatype datatype) {
  switch (datatype) {
    case TensorDatatype::HALF:
      return sizeof(float) / 2;  // NEED CHECK
    case TensorDatatype::FLOAT:
      return sizeof(float);
    case TensorDatatype::DOUBLE:
      return sizeof(double);
    case TensorDatatype::INT32:
      return sizeof(int);
    default:
      return 0;
  }
}

std::string size_in_bytes(int64_t size, TensorDatatype type) {
  return "(" + std::to_string(size) + " * sizeof(" + datatype_string(type) +
         "))";
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

std::string string_gather(std::vector<std::string> &strings,
                          const std::string &delimiter) {
  std::string result;

  for (size_t i = 0; i < strings.size(); ++i) {
    result += strings[i];

    if (i < strings.size() - 1) {
      result += delimiter;
    }
  }

  return result;
}

bool getBoolEnvironmentVariable(const std::string &str, bool default_value) {
  const char *pointer = std::getenv(str.c_str());
  if (pointer == NULL) {
    return default_value;
  }
  std::string value = std::string(pointer);
  std::transform(value.begin(), value.end(), value.begin(), ::toupper);
  return (value == "1" || value == "ON" || value == "YES" || value == "TRUE");
}

int64_t getLevelEnvironmentVariable(const std::string &str,
                                    int64_t default_value) {
  const char *pointer = std::getenv(str.c_str());
  if (pointer == nullptr) {
    return default_value;
  }
  std::string value = std::string(pointer);
  std::transform(value.begin(), value.end(), value.begin(), ::toupper);
  return std::stoll(value);
}

bool ALL(const std::vector<bool> &boolvec) {
  for (size_t i = 0; i < boolvec.size(); i++) {
    if (!boolvec[i]) {
      return false;
    }
  }
  return true;
}

bool ANY(const std::vector<bool> &boolvec) {
  for (size_t i = 0; i < boolvec.size(); i++) {
    if (boolvec[i]) {
      return true;
    }
  }
  return false;
}

}  // namespace infini
