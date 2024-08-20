#ifndef BINARY_MICRO_H
#define BINARY_MICRO_H
#include "core/micro.h"

namespace infini {

#define MAKEOBJ(MICRO)                                                         \
    static Micro *makeObj(const std::vector<Tile *> &inputs,                   \
                          const std::vector<Tile *> &outputs) {                \
        ASSERT(outputs.size() == 1);                                           \
        ASSERT(inputs.size() == 2);                                            \
        return new MICRO(inputs, outputs);                                     \
    }

#define BINARY_DEF(OP, PLATFORM_NAME, PLATFORM)                                \
    class CAT(OP, PLATFORM_NAME) : public BinaryMicro {                        \
      public:                                                                  \
        CAT(OP, PLATFORM_NAME)                                                 \
        (const std::vector<Tile *> &inputs,                                    \
         const std::vector<Tile *> &outputs)                                   \
            : BinaryMicro(inputs, outputs, PLATFORM) {}                        \
        std::string code(Cache &cache, std::string &code,                      \
                         int64_t indent) override;                             \
        MAKEOBJ(CAT(OP, PLATFORM_NAME))                                        \
    }

class BinaryMicro : public Micro {
  protected:
    Tile *left, *right, *output;
    std::string leftName, rightName, outputName;
    int64_t length;
    TensorDataType dataType;

  public:
    BinaryMicro(const std::vector<Tile *> &inputs,
                const std::vector<Tile *> &outputs, Platform platform)
        : Micro(MicroType::BINARY, platform), output(outputs[0]),
          left(inputs[0]), right(inputs[1]),
          outputName(outputs[0]->tensor->tensorName),
          leftName(inputs[0]->tensor->tensorName),
          rightName(inputs[1]->tensor->tensorName),
          length(outputs[0]->getElementNum()),
          dataType(outputs[0]->tensor->tensorDataType) {}
    virtual std::string code(Cache &cache, std::string &code,
                             int64_t indent = 0) = 0;
    static Micro *makeObj() { return nullptr; }
};

/**
 * Cuda Micro declearation
 *  1. AddCuda
 *  2. SubCuda
 *  3. MulCuda
 *  4. DivCuda
 *  5. EqCuda
 *  6. GeCuda
 *  7. GtCuda
 *  8. LeCuda
 *  9. LtCuda
 * 10. NeCuda
 * 11. AndCuda
 * 12. OrCuda
 * 13. XorCuda
 * 14. FloorModCuda
 * 15. FloorDivCuda
 */
BINARY_DEF(Add, Cuda, Platform::CUDA);
BINARY_DEF(Sub, Cuda, Platform::CUDA);
BINARY_DEF(Mul, Cuda, Platform::CUDA);
BINARY_DEF(Div, Cuda, Platform::CUDA);
BINARY_DEF(Eq, Cuda, Platform::CUDA);
BINARY_DEF(Ge, Cuda, Platform::CUDA);
BINARY_DEF(Gt, Cuda, Platform::CUDA);
BINARY_DEF(Le, Cuda, Platform::CUDA);
BINARY_DEF(Lt, Cuda, Platform::CUDA);
BINARY_DEF(Ne, Cuda, Platform::CUDA);
BINARY_DEF(And, Cuda, Platform::CUDA);
BINARY_DEF(Or, Cuda, Platform::CUDA);
BINARY_DEF(Xor, Cuda, Platform::CUDA);
BINARY_DEF(FloorMod, Cuda, Platform::CUDA);
BINARY_DEF(FloorDiv, Cuda, Platform::CUDA);

/**
 * Bang Micro declaration
 *  1. AddBang
 *  2. SubBang
 *  3. MulBang
 *  4. DivBang
 *  5. EqBang
 *  6. GeBang
 *  7. GtBang
 *  8. LeBang
 *  9. LtBang
 * 10. NeBang
 * 11. AndBang
 * 12. OrBang
 * 13. XorBang
 */
BINARY_DEF(Add, Bang, Platform::BANG);
BINARY_DEF(Sub, Bang, Platform::BANG);
BINARY_DEF(Mul, Bang, Platform::BANG);
BINARY_DEF(Div, Bang, Platform::BANG);
BINARY_DEF(Eq, Bang, Platform::BANG);
BINARY_DEF(Ge, Bang, Platform::BANG);
BINARY_DEF(Gt, Bang, Platform::BANG);
BINARY_DEF(Le, Bang, Platform::BANG);
BINARY_DEF(Lt, Bang, Platform::BANG);
BINARY_DEF(Ne, Bang, Platform::BANG);
BINARY_DEF(And, Bang, Platform::BANG);
BINARY_DEF(Or, Bang, Platform::BANG);
BINARY_DEF(Xor, Bang, Platform::BANG);
BINARY_DEF(FloorMod, Bang, Platform::BANG);
BINARY_DEF(FloorDiv, Bang, Platform::BANG);

#undef MAKEOBJ
#undef BINARY_DEF
} // namespace infini
#endif
