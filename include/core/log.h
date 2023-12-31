#pragma once
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#define LOG_PURE 0
#define LOG_INFO 1
#define LOG_WARNING 2
#define LOG_ERROR 3
#define LOG_FATAL 4
#define LOG_DLOG 5

#define PRINTLOG(name, severity) \
  infini::Log(__FILE__, __LINE__, LOG_##severity, 0, #name).stream()

#define DEVELOPLOG(name, level) \
  infini::Log(__FILE__, __LINE__, LOG_DLOG, level, #name).stream()

namespace infini {

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
  // 构造与析构
  Log() = delete;
  Log(std::string file, int32_t line, int32_t severity, int32_t module,
      std::string name);
  ~Log();
  // 时间
  std::string getTime();
  // 基本信息
  void printHead();
  void printTail();
  // 日志信息
  std::stringstream &stream();
};

bool getBoolEnvironment(const std::string &str, bool default_value);

int32_t getLevelEnvironment(const std::string &str, int32_t default_value);

}  // namespace infini
