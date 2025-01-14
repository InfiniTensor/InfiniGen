#ifndef UNARY_MICRO_H
#define UNARY_MICRO_H
#include "core/micro.h"

namespace infini {

#define MAKEOBJ(MICRO)                                                         \
    static Micro *makeObj(const std::vector<Tile *> &inputs,                   \
                          const std::vector<Tile *> &outputs) {                \
        ASSERT(outputs.size() == 1);                                           \
        ASSERT(inputs.size() == 1);                                            \
        return new MICRO(inputs, outputs);                                     \
    }

#define UNARY_DEF(OP, PLATFORM_NAME, PLATFORM)                                 \
    class CAT(OP, PLATFORM_NAME) : public UnaryMicro {                         \
      public:                                                                  \
        CAT(OP, PLATFORM_NAME)                                                 \
        (const std::vector<Tile *> &inputs,                                    \
         const std::vector<Tile *> &outputs)                                   \
            : UnaryMicro(inputs, outputs, PLATFORM) {}                         \
        std::string code(Cache &cache, std::string &code,                      \
                         int64_t indent) override;                             \
        MAKEOBJ(CAT(OP, PLATFORM_NAME))                                        \
    }

class UnaryMicro : public Micro {
  protected:
    Tile *input, *output;
    std::string inputName, outputName;
    int64_t length;
    TensorDataType dataType;

  public:
    UnaryMicro(const std::vector<Tile *> &inputs,
               const std::vector<Tile *> &outputs, Platform platform)
        : Micro(MicroType::UNARY, platform), input(inputs[0]),
          output(outputs[0]), inputName(inputs[0]->tensor->tensorName),
          outputName(outputs[0]->tensor->tensorName),
          length(outputs[0]->getElementNum()),
          dataType(outputs[0]->tensor->tensorDataType) {}
    virtual std::string code(Cache &cache, std::string &code,
                             int64_t indent = 0) = 0;
    static Micro *makeObj() { return nullptr; }
};

/**
 * Cuda Unary micros
 *  1. SqrtCuda
 *  2. SigmoidCuda
 *  3. SoftmaxCuda
 *  4. ReluCuda
 */
UNARY_DEF(Sqrt, Cuda, Platform::CUDA);
UNARY_DEF(Sigmoid, Cuda, Platform::CUDA);
UNARY_DEF(Relu, Cuda, Platform::CUDA);
UNARY_DEF(RSqrt, Cuda, Platform::CUDA);
UNARY_DEF(Recip, Cuda, Platform::CUDA);
UNARY_DEF(Sin, Cuda, Platform::CUDA);
UNARY_DEF(Cos, Cuda, Platform::CUDA);
UNARY_DEF(Tanh, Cuda, Platform::CUDA);

/**
 * Bang Unary micros
 *  1. SqrtBang
 *  2. SigmoidBang
 *  3. SoftmaxBang
 *  4. ReluBang
 */
UNARY_DEF(Sqrt, Bang, Platform::BANG);
UNARY_DEF(Sigmoid, Bang, Platform::BANG);
UNARY_DEF(Relu, Bang, Platform::BANG);
UNARY_DEF(RSqrt, Bang, Platform::BANG);
UNARY_DEF(Recip, Bang, Platform::BANG);
UNARY_DEF(Cos, Bang, Platform::BANG);
UNARY_DEF(Sin, Bang, Platform::BANG);
UNARY_DEF(Tanh, Bang, Platform::BANG);

UNARY_DEF(Sqrt, Ascend, Platform::ASCEND);
UNARY_DEF(Sigmoid, Ascend, Platform::ASCEND);
UNARY_DEF(Relu, Ascend, Platform::ASCEND);
UNARY_DEF(RSqrt, Ascend, Platform::ASCEND);
UNARY_DEF(Recip, Ascend, Platform::ASCEND);
UNARY_DEF(Cos, Ascend, Platform::ASCEND);
UNARY_DEF(Sin, Ascend, Platform::ASCEND);
UNARY_DEF(Tanh, Ascend, Platform::ASCEND);
UNARY_DEF(Abs, Ascend, Platform::ASCEND);

UNARY_DEF(Sqrt, Kunlun, Platform::KUNLUN);
UNARY_DEF(Sigmoid, Kunlun, Platform::KUNLUN);
UNARY_DEF(Relu, Kunlun, Platform::KUNLUN);
UNARY_DEF(RSqrt, Kunlun, Platform::KUNLUN);
UNARY_DEF(Recip, Kunlun, Platform::KUNLUN);
UNARY_DEF(Cos, Kunlun, Platform::KUNLUN);
UNARY_DEF(Sin, Kunlun, Platform::KUNLUN);
UNARY_DEF(Tanh, Kunlun, Platform::KUNLUN);
UNARY_DEF(Abs, Kunlun, Platform::KUNLUN);

#undef MAKEOBJ
#undef UNARY_DEF

} // namespace infini
#endif
