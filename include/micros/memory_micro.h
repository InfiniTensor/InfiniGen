#ifndef MEMORY_MICRO_H
#define MEMORY_MICRO_H
#include "core/micro.h"

namespace infini {

#define MAKEOBJ(MICRO)                                                         \
    static Micro *makeObj(const std::vector<Tile *> &inputs,                   \
                          const std::vector<Tile *> &outputs = {}) {           \
        ASSERT(inputs.size() == 1);                                            \
        return new MICRO(inputs, outputs);                                     \
    }

#define MEMORY_DEF(OP, PLATFORM_NAME, PLATFORM)                                \
    class CAT(OP, PLATFORM_NAME) : public MemoryMicro {                        \
      public:                                                                  \
        CAT(OP, PLATFORM_NAME)                                                 \
        (const std::vector<Tile *> &inputs,                                    \
         const std::vector<Tile *> &outputs = {})                              \
            : MemoryMicro(inputs, outputs, PLATFORM) {}                        \
        std::string code(Cache &cache, std::string &code,                      \
                         int64_t indent) override;                             \
        MAKEOBJ(CAT(OP, PLATFORM_NAME))                                        \
    }

class MemoryMicro : public Micro {
  protected:
    Tile *operand;
    std::string name;
    int64_t length;
    TensorDataType dataType;

  public:
    MemoryMicro(const std::vector<Tile *> &inputs,
                const std::vector<Tile *> &outputs, Platform platform)
        : Micro(MicroType::MEMORY, platform), operand(inputs[0]),
          name(inputs[0]->tensor->tensorName),
          length(inputs[0]->getElementNum()),
          dataType(inputs[0]->tensor->tensorDataType) {}
    virtual std::string code(Cache &cache, std::string &code,
                             int64_t indent = 0) = 0;
    static Micro *makeObj() { return nullptr; }
};

/**
 * Cuda memory micro implementation, including
 *  1. LoadCuda
 *  2. AllocateCuda
 *  3. StoreCuda
 *  4. FreeCuda
 */
MEMORY_DEF(Load, Cuda, Platform::CUDA);
MEMORY_DEF(Allocate, Cuda, Platform::CUDA);
MEMORY_DEF(Store, Cuda, Platform::CUDA);
MEMORY_DEF(Free, Cuda, Platform::CUDA);

/**
 * Bang memory micro implementation, including
 *  1. LoadBang
 *  2. AllocateBang
 *  3. StoreBang
 *  4. FreeBang
 */
MEMORY_DEF(Load, Bang, Platform::BANG);
MEMORY_DEF(Allocate, Bang, Platform::BANG);
MEMORY_DEF(Store, Bang, Platform::BANG);
MEMORY_DEF(Free, Bang, Platform::BANG);

/**
 * Ascend memory micro implementation, including
 *  1. LoadAscend
 *  2. AllocateAscend
 *  3. StoreAscend
 *  4. FreeAscend
 */
MEMORY_DEF(Load, Ascend, Platform::ASCEND);
MEMORY_DEF(Allocate, Ascend, Platform::ASCEND);
MEMORY_DEF(Store, Ascend, Platform::ASCEND);
MEMORY_DEF(Free, Ascend, Platform::ASCEND);

#undef MAKEOBJ
#undef MEMORY_DEF
} // namespace infini
#endif
