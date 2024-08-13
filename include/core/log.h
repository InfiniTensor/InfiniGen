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