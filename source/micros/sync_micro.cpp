#include "micros/sync_micro.h"
#include "core/cache.h"

namespace infini {

std::string BangSyncMicro::generatorCode(Cache& cache, std::string& code,
                                         int64_t indent) {
  code += "__sync_all();\n";
  return "";
}

std::string CudaSyncMicro::generatorCode(Cache& cache, std::string& code,
                                         int64_t indent) {
  code += "TODO;\n";
  return "";
}

}  // namespace infini
