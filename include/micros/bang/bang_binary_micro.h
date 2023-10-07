#pragma once
#include "micros/binary_micro.h"

namespace infini {

BINARY_MICRO(Bang, Add, MicroType::BINARY, Platform::BANG)
BINARY_MICRO(Bang, Sub, MicroType::BINARY, Platform::BANG)
BINARY_MICRO(Bang, Mul, MicroType::BINARY, Platform::BANG)

REGISTER_MICRO_CONSTRUCTOR(OperatorType::ADD, Platform::BANG, AddBang::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::SUB, Platform::BANG, SubBang::makeObj)
REGISTER_MICRO_CONSTRUCTOR(OperatorType::MUL, Platform::BANG, MulBang::makeObj)
}  // namespace infini