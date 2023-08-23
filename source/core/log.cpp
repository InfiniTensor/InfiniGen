#include "core/log.h"

#define RESET "\033[0m"
#define HIGHLIGHT "\033[1m"
#define UNDERLINE "\033[4m"
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"

namespace infini {

Log::Log(std::string file, int32_t line, int32_t severity, int32_t module,
         std::string name) {
  log_file = file;
  log_line = line;
  log_severity = severity;
  log_module = module;
  module_name = name;
}

Log::~Log() {
  if (log_severity != LOG_PURE) {
    printHead();
  }
  if (log_severity != LOG_INFO && log_severity != LOG_PURE) {
    printTail();
  }
  file_string << context_string.str();
  print_string << context_string.str();
  print_stream << print_string.str();
  if (log_severity != LOG_PURE) {
    print_stream << std::endl;
  }
}

std::string Log::getTime() { return ""; }

void Log::printHead() {
  switch (log_severity) {
    case LOG_PURE:
      file_string << "[PURE]: ";
      print_string << HIGHLIGHT << GREEN << "[PURE]: " << RESET;
      break;
    case LOG_INFO:
      file_string << "[INFO]: " << module_name << " ";
      print_string << HIGHLIGHT << GREEN << "[INFO]: " << module_name << " "
                   << RESET;
      break;
    case LOG_WARNING:
      file_string << "[WARNING]: " << module_name << " ";
      print_string << HIGHLIGHT << YELLOW << "[WARNING]: " << module_name << " "
                   << RESET;
      break;
    case LOG_ERROR:
      file_string << "[ERROR]: " << module_name << " ";
      print_string << HIGHLIGHT << RED << "[ERROR]: " << module_name << " "
                   << RESET;
      break;
    case LOG_FATAL:
      file_string << "[FATAL]: " << module_name << " ";
      print_string << HIGHLIGHT << RED << "[FATAL]: " << module_name << " "
                   << RESET;
      break;
    default:
      break;
  }
}

void Log::printTail() {
  file_string << log_file << ":" << log_line << "  ";
  print_string << GREEN << log_file << ":" << log_line << "  " << RESET;
}

std::stringstream& Log::stream() { return context_string; }

}  // namespace infini
