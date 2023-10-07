#pragma once
#include "micros/binary_micro.h"

namespace infini {

BINARY_MICRO(Cuda, Add, MicroType::BINARY, Platform::CUDA)
BINARY_MICRO(Cuda, Sub, MicroType::BINARY, Platform::CUDA)
BINARY_MICRO(Cuda, Mul, MicroType::BINARY, Platform::CUDA)

REGISTER_MICRO_CONSTRUCTOR(OperatorType::ADD, Platform::CUDA, AddCuda::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::SUB, Platform::CUDA, SubCuda::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::MUL, Platform::CUDA, MulCuda::makeObj)
}  // namespace infini
