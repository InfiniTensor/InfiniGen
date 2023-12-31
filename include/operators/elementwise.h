#pragma once
#include "core/operator.h"
#include "kernels/binary.h"
#include "kernels/memory.h"

namespace infini {

class Binary : public Operator {
 public:
  // Constructor
  Binary(OperatorType type, Tensor* input_left, Tensor* input_right,
         Tensor* output);
  // Destructor
  ~Binary();
  // Apply
  void applySplit() override;
  // Generator
  // std::string generatorBone(PlatformType platform) override;

 private:
  // Check
  bool checkValid() override;
  // std::string generatorBoneOnCUDA(std::string name);
  // std::string generatorBoneOnBANG(std::string name);
  // std::string generatorCoreOnCUDA(const std::vector<int64_t>& coord);
  // std::string generatorCoreOnBANG(const std::vector<int64_t>& coord);
};

#define DEFINE_BINARY(OP_NAME)                                             \
  class OP_NAME : public Binary {                                          \
   public:                                                                 \
    OP_NAME(Tensor* input_left, Tensor* input_right, Tensor* output)       \
        : Binary(OperatorType::OP_NAME, input_left, input_right, output) { \
      Kernel* kernel = new G2R##Kernel();                                  \
      this->pushKernel(kernel);                                            \
      kernel = new G2R##Kernel();                                          \
      this->pushKernel(kernel);                                            \
      kernel = new OP_NAME##Kernel();                                      \
      this->pushKernel(kernel);                                            \
      kernel = new R2G##Kernel();                                          \
      this->pushKernel(kernel);                                            \
    }                                                                      \
  };

DEFINE_BINARY(ADD)
DEFINE_BINARY(SUB)
DEFINE_BINARY(MUL)
#undef DEFINE_BINARY_OBJ

}  // namespace infini
