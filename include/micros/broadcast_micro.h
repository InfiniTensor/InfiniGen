#ifndef BROADCAST_MICRO_H
#define BROADCAST_MICRO_H
#include "core/micro.h"

namespace infini {

#define MAKEOBJ(MICRO)                                                         \
    static Micro *makeObj(const std::vector<Tile *> &inputs,                   \
                          const std::vector<Tile *> &outputs) {                \
        ASSERT(outputs.size() == 1);                                           \
        ASSERT(inputs.size() == 1);                                            \
        return new MICRO(inputs, outputs);                                     \
    }

#define BROADCAST_DEF(PLATFORM_NAME, PLATFORM)                                 \
    class Broadcast##PLATFORM_NAME : public BroadcastMicro {                   \
      public:                                                                  \
        Broadcast##PLATFORM_NAME(const std::vector<Tile *> &inputs,            \
                                 const std::vector<Tile *> &outputs = {})      \
            : BroadcastMicro(inputs, outputs, PLATFORM) {}                     \
        std::string code(Cache &cache, std::string &code,                      \
                         int64_t indent) override;                             \
        MAKEOBJ(Broadcast##PLATFORM_NAME)                                      \
    }

class BroadcastMicro : public Micro {
  protected:
    Tile *input, *output;
    std::string inputName, outputName;
    int64_t length;
    TensorDataType dataType;

  public:
    BroadcastMicro(const std::vector<Tile *> &inputs,
                   const std::vector<Tile *> &outputs, Platform platform)
        : Micro(MicroType::BROADCAST, platform), input(inputs[0]),
          output(outputs[0]), inputName(inputs[0]->tensor->tensorName),
          outputName(outputs[0]->tensor->tensorName),
          length(outputs[0]->getElementNum()),
          dataType(outputs[0]->tensor->tensorDataType) {}
    virtual std::string code(Cache &cache, std::string &code,
                             int64_t indent = 0) = 0;
    static Micro *makeObj() { return nullptr; }
};

BROADCAST_DEF(Bang, Platform::BANG);
BROADCAST_DEF(Cuda, Platform::CUDA);

#undef MAKEOBJ
#undef BROADCAST_DEF

} // namespace infini
#endif
