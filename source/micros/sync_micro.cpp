#include "micros/sync_micro.h"
#include "core/cache.h"

namespace infini {

std::string SyncBang::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  code += "__sync_all();\n";
  return "";
}

std::string SyncCuda::generatorCode(Cache& cache, std::string& code,
                                    int64_t indent) {
  code += "TODO;\n";
  return "";
}

}  // namespace infini
