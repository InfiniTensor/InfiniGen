#include "core/utils.h"

std::ofstream &LOG_FILE(std::string file_path) {
    infini::log_stream.flush();
    infini::log_stream.close();
    infini::log_stream.open(file_path, std::ios::out);
    return infini::log_stream;
}

namespace infini {

void COMPILE(std::string input_file_path, std::string output_binary_directory,
             Platform platform) {
    auto file_path_split = STRING_SPLIT(input_file_path, '/');
    std::string file = file_path_split[file_path_split.size() - 1];
    std::string file_name = STRING_SPLIT(file, '.')[0];
    std::string shell = "";
    if (platform == Platform::BANG) {
        shell += "cncc -shared -fPIC -o " + output_binary_directory + "lib" +
                 file_name + ".so " + input_file_path +
                 " --bang-mlu-arch=mtp_592 -O3";
        system(shell.c_str());
    } else if (platform == Platform::CUDA) {
        shell += "nvcc -arch=sm_80 -shared --compiler-options '-fPIC' -o " +
                 output_binary_directory + "lib" + file_name + ".so " +
                 input_file_path + " -O3 --extended-lambda";
        system(shell.c_str());
    }
    return;
}

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
    default:
        return "UNKNOWN";
    }
}

std::string dataTypeStr(TensorDataType datatype) {
    switch (datatype) {
    case TensorDataType::CHAR:
        return "char";
    case TensorDataType::HALF:
        return "half";
    case TensorDataType::FLOAT:
        return "float";
    case TensorDataType::DOUBLE:
        return "double";
    default:
        return "UNKNOWN";
    }
}

std::string TO_STRING(OperatorType type) {
#define CASE(NAME)                                                             \
    case OperatorType::NAME:                                                   \
        return #NAME
    switch (type) {
        CASE(ADD);
        CASE(SUB);
        CASE(MUL);
        CASE(DIV);
        CASE(EQ);
        CASE(GE);
        CASE(GT);
        CASE(LE);
        CASE(LT);
        CASE(NE);
        CASE(AND);
        CASE(OR);
        CASE(XOR);
        CASE(SQRT);
        CASE(RSQRT);
        CASE(RECIP);
        CASE(SIGMOID);
        CASE(RELU);
        CASE(SIN);
        CASE(COS);
        CASE(TANH);
        CASE(LOAD);
        CASE(ALLOCATE);
        CASE(FREE);
        CASE(STORE);
        CASE(BROADCAST);
        CASE(BROADCAST_ADD);
        CASE(BROADCAST_SUB);
        CASE(BROADCAST_MUL);
        CASE(BROADCAST_DIV);
        CASE(REDUCE);
        CASE(SYNC);
    default:
        return "UNKNOWN";
    }
#undef CASE
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

std::string INITIALIZER(const std::vector<int64_t> &input) {
    std::string info_string = "{";
    for (auto i = 0; i < input.size(); ++i) {
        info_string += std::to_string(input[i]);
        info_string += (i == (input.size() - 1) ? "" : ", ");
    }
    info_string += "}";
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

std::string TO_STRING(MicroType type) {
#define CASE(NAME)                                                             \
    case MicroType::NAME:                                                      \
        return #NAME
    switch (type) {
        CASE(BINARY);
        CASE(UNARY);
        CASE(REDUCE);
        CASE(BROADCAST);
        CASE(MEMORY);
    default:
        return "UNKNOWN";
    }
#undef CASE
}

std::string TO_STRING(CachePolicy policy) {
#define CASE(NAME)                                                             \
    case CachePolicy::NAME:                                                    \
        return #NAME
    switch (policy) {
        CASE(LRU);
        CASE(LFU);
        CASE(FIFO);
        CASE(DEFAULT);
    default:
        return "UNKNOWN";
    }
#undef CASE
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

std::string INDENTATION(int64_t num) { return std::string(num * 2, ' '); }

std::string STRING_GATHER(std::vector<std::string> &strings,
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

} // namespace infini
