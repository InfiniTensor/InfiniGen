#pragma once
#include "micros/memory_micro.h"

namespace infini {

MEMORY_MICRO(Cuda, Load, MicroType::MEMORY, Platform::CUDA)
MEMORY_MICRO(Cuda, Store, MicroType::MEMORY, Platform::CUDA)
MEMORY_MICRO(Cuda, Allocate, MicroType::MEMORY, Platform::CUDA)
MEMORY_MICRO(Cuda, Free, MicroType::MEMORY, Platform::CUDA)

REGISTER_MICRO_CONSTRUCTOR(OperatorType::LOAD, Platform::CUDA,
                           LoadCuda::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::STORE, Platform::CUDA,
                           StoreCuda::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::ALLOCATE, Platform::CUDA,
                           AllocateCuda::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::FREE, Platform::CUDA,
                           FreeCuda::makeObj)

}  // namespace infini
