#pragma once
#include "core/micro.h"

namespace infini {

#define MAKEOBJ(MICRO)                                              \
  static Micro* makeObj(const std::vector<OperandType>& operands) { \
    ASSERT(operands.empty());                                       \
    return new MICRO();                                             \
  }

#define SYNC_DEF(OP, PLName, PL)                               \
  class CAT(OP, PLName) : public SyncMicro {                   \
   public:                                                     \
    CAT(OP, PLName)() : SyncMicro(PL) {}                       \
    std::string generatorCode(Cache& cache, std::string& code, \
                              int64_t indent) override;        \
    MAKEOBJ(CAT(OP, PLName))                                   \
  }

class SyncMicro : public Micro {
 public:
  SyncMicro(Platform pt) : Micro(MicroType::SYNC, pt) {}
};

/**
 * Cuda Sync declearation, including
 *  1. SyncCuda
 */
SYNC_DEF(Sync, Cuda, Platform::CUDA);

/**
 * Bang Sync declearation, including
 *  1. SyncBang
 */
SYNC_DEF(Sync, Bang, Platform::BANG);

#undef MAKEOBJ
#undef SYNC_DEF
}  // namespace infini
