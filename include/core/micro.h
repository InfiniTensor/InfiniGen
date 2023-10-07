#pragma once
#include "core/type.h"
#include "core/cache.h"
#include "core/platform.h"
#include "core/utils.h"
#include <string>
#include <functional>
#include <map>
#include <tuple>

namespace infini {

// TODO: delete line after change to Kernel
using MicroType = KernelType;

using OperandType =
    std::tuple<std::string, int, int>;  // operand_name, offset, length

class Micro {
  /**
   * @brief Micro is smallest unit of codegen.
   *  It includes different types of memory access,
   *  operation codegen, and others
   */
 protected:
  MicroType micro_type;
  Platform platform;

 public:
  // Constructor
  Micro(const Micro &) = delete;
  Micro() = default;
  Micro(MicroType mt, Platform pt) : micro_type(mt), platform(pt) {}
  // Destructor
  virtual ~Micro(){};
  /**
   * @brief Generate code. Subclasses rewrite this
   * function and call generations which belongs to
   * specific platform.
   */
  virtual std::string generatorCode(Cache &cache, std::string &result,
                                    int64_t indent = 0) = 0;
  static Micro *makeObj();  // dummpy functon
  /** @brief Information print*/
  virtual void printInformation();
};

using MicroAttrs = std::tuple<OperatorType, Platform>;
using MicroConstructor =
    std::function<Micro *(const std::vector<OperandType> &)>;

class MicroRegistry {
 private:
  std::map<MicroAttrs, MicroConstructor> microrecords;
  int nrecord = 0;

 public:
  ~MicroRegistry() = default;

  static MicroRegistry &getInstance() {
    static MicroRegistry instance;
    return instance;
  }
  bool registerMicro(const MicroAttrs &key, MicroConstructor constructor) {
    ASSERT(microrecords.find(key) == microrecords.end());
    microrecords.emplace(key, constructor);
    nrecord++;
    return true;
  }
  const MicroConstructor &getConstructor(const MicroAttrs &key) const {
    return microrecords.at(key);
  }
};

#define REGISTER_MICRO_CONSTRUCTOR(optype, platform, constructor)              \
  static const bool _register_micro_constructor_##__COUNTER__ =                \
      MicroRegistry::getInstance().registerMicro(MicroAttrs{optype, platform}, \
                                                 constructor);

}  // namespace infini
