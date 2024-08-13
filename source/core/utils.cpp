#include "core/utils.h"

std::ofstream &LOG_FILE(std::string file_path) {
    infini::log_stream.flush();
    infini::log_stream.close();
    infini::log_stream.open(file_path, std::ios::out);
    return infini::log_stream;
}

namespace infini {

bool ANY_TRUE(const std::vector<bool> &input) {
    for (size_t i = 0; i < input.size(); i++) {
        if (input[i]) {
            return true;
        }
    }
    return false;
}

bool ALL_TRUE(const std::vector<bool> &input) {
    for (size_t i = 0; i < input.size(); i++) {
        if (!input[i]) {
            return false;
        }
    }
    return true;
}

std::string TO_STRING(TensorDataType datatype) {
    switch (datatype) {
    case TensorDataType::CHAR:
        return "CHAR";
    case TensorDataType::HALF:
        return "HALF";
    case TensorDataType::FLOAT:
        return "FLOAT";
    case TensorDataType::DOUBLE:
        return "DOUBLE";
    case TensorDataType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

std::string TO_STRING(const std::vector<int64_t> &input) {
    std::string info_string = "[";
    for (auto i = 0; i < input.size(); ++i) {
        info_string += std::to_string(input[i]);
        info_string += (i == (input.size() - 1) ? "" : ", ");
    }
    info_string += "]";
    return info_string;
}

std::string TO_STRING(const std::vector<std::string> &input) {
    std::string info_string = "[";
    for (auto i = 0; i < input.size(); ++i) {
        info_string += input[i];
        info_string += (i == (input.size() - 1) ? "" : ", ");
    }
    info_string += "]";
    return info_string;
}

std::string TO_STRING(const bool input) {
    switch (input) {
    case false:
        return "FALSE";
    case true:
        return "TRUE";
    default:
        return "UNKNOWN";
    }
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

int64_t VECTOR_DOT_PRODUCT(const std::vector<int64_t> &left,
                           const std::vector<int64_t> &right) {
    ASSERT(left.size() == right.size());
    int64_t result = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        result += left[i] * right[i];
    }
    return result;
}

std::vector<int64_t> MINIMUM(const std::vector<int64_t> &left,
                             const std::vector<int64_t> &right) {
    ASSERT(left.size() == right.size());
    std::vector<int64_t> result;
    for (size_t i = 0; i < left.size(); ++i) {
        result.push_back(std::min(left[i], right[i]));
    }
    return result;
}

std::vector<int64_t> MAXIMUM(const std::vector<int64_t> &left,
                             const std::vector<int64_t> &right) {
    ASSERT(left.size() == right.size());
    std::vector<int64_t> result;
    for (size_t i = 0; i < left.size(); ++i) {
        result.push_back(std::max(left[i], right[i]));
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

int64_t SIZE_OF(TensorDataType datatype) {
    switch (datatype) {
    case TensorDataType::CHAR:
        return 1;
    case TensorDataType::HALF:
        return 2;
    case TensorDataType::FLOAT:
        return 4;
    case TensorDataType::DOUBLE:
        return 8;
    case TensorDataType::UNKNOWN:
    default:
        return 0;
    }
}

std::vector<int64_t> CALCULATE_STRIDE(const std::vector<int64_t> &shape) {
    std::vector<int64_t> result(shape.size());
    int value = 1;
    for (int i = 0; i < shape.size(); i++) {
        result[shape.size() - 1 - i] = value;
        value *= shape[shape.size() - 1 - i];
    }
    return result;
}

} // namespace infini