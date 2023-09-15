#pragma
#include "core/api.h"
#include "core/cache.h"
#include "core/micro.h"
#include "micros/cuda/cuda_micro.h"

namespace infini {

// TODO: delete this line after change name
using MicroType = KernelType;

#define BINARY_CUDA_MICRO(prefix, MICRO_TYPE)       \
  class prefix##CudaMicro : public CudaMicro {      \
    int64_t output, left, right, length;            \
    std::string output_name, left_name, right_name; \
                                                    \
   public:                                          \
    prefix##CudaMicro
}  // namespace infini

CUDA_MICRO(Add, MicroType::ADD)
CUDA_MICRO(Sub, MicroType::SUB)
CUDA_MICRO(Mul, MicroType::MUL)

}  // namespace infini
