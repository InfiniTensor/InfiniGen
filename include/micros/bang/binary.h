#pragma
#include "core/api.h"
#include "core/micro.h"
#include "micros/bang/bang_micro.h"

namespace infini {

// TODO: delete this line after change to Kernel
using KernelType = MicroType;

BANG_MICRO(Add, MicroType::ADD)
BANG_MICRO(Sub, MicroType::SUB)
BANG_MICRO(Mul, MicroType::MUL)

}  // namespace infini
