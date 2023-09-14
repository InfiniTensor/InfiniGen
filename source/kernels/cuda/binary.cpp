#include "core/api.h"
#include "kernels/cuda/cuda_kernel.h"
#include "kernels/cuda/binary.h"
#include <vector>

using std::string;
using std::vector;

/**
 * implementation of binary kernel
 * 
*/
namespace infini {

  string ADDCudaKernel::generateCodeOnCuda(vector<string>& args) const {
    return args[2] + " = " + args[0] + " + " + args[1]
  }

  string SUBCudaKernel::generateCodeOnCuda(vector<string>& args) const {

  }

  string MULCudaKernel::generateCodeOnCuda(vector<string>& args) const {

  }
  

}  // namespace infini