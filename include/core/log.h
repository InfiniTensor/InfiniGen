#ifndef LOG_H
#define LOG_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define LOG_PURE 0
#define LOG_INFO 1
#define LOG_WARNING 2
#define LOG_ERROR 3
#define LOG_FATAL 4
#define LOG_DLOG 5

#define RESET "\033[0m"
#define HIGHLIGHT "\033[1m"
#define UNDERLINE "\033[4m"
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"

#define BRIGHT_BLACK "\033[90m"
#define BRIGHT_RED "\033[91m"
#define BRIGHT_GREEN "\033[92m"
#define BRIGHT_YELLOW "\033[93m"
#define BRIGHT_BLUE "\033[94m"
#define BRIGHT_MAGENTA "\033[95m"
#define BRIGHT_CYAN "\033[96m"
#define BRIGHT_WHITE "\033[97m"

#define PRINTLOG(name, severity)                                               \
    infini::Log(__FILE__, __LINE__, LOG_##severity, 0, #name).stream()

#define DEVELOPLOG(name, level)                                                \
    infini::Log(__FILE__, __LINE__, LOG_DLOG, level, #name).stream()

namespace infini {

static std::ofstream log_stream;

class Log {
  public:
    std::string log_file;
    int32_t log_line;
    int32_t log_severity;
    int32_t log_module;
    std::string module_name;
    std::stringstream context_string;
    std::stringstream print_string;
    std::stringstream file_string;

  public:
    Log() = delete;
    Log(std::string file, int32_t line, int32_t severity, int32_t module,
        std::string name);
    ~Log();

    std::string getTime();

    void printHead();
    void printTail();

    std::stringstream &stream();
};

bool getBoolEnvironment(const std::string &str, bool default_value);

int32_t getLevelEnvironment(const std::string &str, int32_t default_value);

} // namespace infini

#endif