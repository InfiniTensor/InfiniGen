#include "micros/bang/bang_binary_micro.h"
#include "micros/bang/bang_memery_micro.h"

namespace infini {

#define BANG_BINARY_IMPLEMENT(OP, OP_STRING)                                 \
  std::string OP##Bang::generatorCode(Cache& cache, std::string& code,       \
                                      int64_t indent) {                      \
    cache.lock();                                                            \
    std::string left_cache =                                                 \
        LoadBang(std::tuple(left_name, left_offset, length))                 \
            .generatorCode(cache, code, indent);                             \
    std::string right_cache =                                                \
        LoadBang(std::tuple(right_name, right_offset, length))               \
            .generatorCode(cache, code, indent);                             \
    std::string output_cache =                                               \
        AllocateBang(std::tuple(output_name, output_offset, length))         \
            .generatorCode(cache, code, indent);                             \
    code += indentation(indent) + "__bang_" + std::string(OP_STRING) + "(" + \
            output_cache + ", " + left_cache + ", " + right_cache + ", " +   \
            std::to_string(length) + ");\n";                                 \
    cache.unlock();                                                          \
    return "";                                                               \
  }

BANG_BINARY_IMPLEMENT(Add, "add")
BANG_BINARY_IMPLEMENT(Sub, "sub")
BANG_BINARY_IMPLEMENT(Mul, "mul")

#undef BANG_BINARY_IMPLEMENT
}  // namespace infini