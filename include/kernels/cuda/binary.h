#pragma once
#include "kernels/cuda/cuda_kernel.h"
#include "core/type.h"

namespace infini {
  
CUDA_KERNEL(ADD, KernelType::ADD)
CUDA_KERNEL(SUB, KernelType::SUB)
CUDA_KERNEL(MUL, KernelType::MUL)

}  // namespace infini
