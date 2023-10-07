#pragma once
#include "micros/memory_micro.h"

namespace infini {

MEMORY_MICRO(Bang, Load, MicroType::MEMORY, Platform::BANG)
MEMORY_MICRO(Bang, Store, MicroType::MEMORY, Platform::BANG)
MEMORY_MICRO(Bang, Allocate, MicroType::MEMORY, Platform::BANG)
MEMORY_MICRO(Bang, Free, MicroType::MEMORY, Platform::BANG)

REGISTER_MICRO_CONSTRUCTOR(OperatorType::LOAD, Platform::BANG,
                           LoadBang::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::STORE, Platform::BANG,
                           StoreBang::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::ALLOCATE, Platform::BANG,
                           AllocateBang::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::FREE, Platform::BANG,
                           FreeBang::makeObj)

}  // namespace infini