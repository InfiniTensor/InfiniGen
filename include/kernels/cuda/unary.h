#pragma
#include "core/type.h"
#include "kernels/cuda/cuda_kernel.h"


namespace infini{
CUDA_KERNEL(SIN, KernelType::SIN)
CUDA_KERNEL(COS, KernelType::COS)    
} // namespace infini
