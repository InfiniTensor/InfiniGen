#include "core/split.h"
#include "core/utils.h"

namespace infini {

Split::Split(const std::vector<int64_t>& dimension) {
  split_dimension = dimension;
}

void Split::printInformation() {
  std::string info_string = "";
  info_string += "—— Split ";
  info_string += "Dimension: ";
  info_string += TO_STRING(split_dimension);
  LOG(INFO) << info_string;
}

void Split::printSummary() {
  std::string info_string = "";
  info_string += "Split ";
  info_string += "Dim: ";
  info_string += TO_STRING(split_dimension);
  info_string += "\n";
  LOG(PURE) << info_string;
}

}  // namespace infini
