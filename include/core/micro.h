#pragma once
#include "core/type.h"
#include "core/utils.h"
#include "core/cache.h"
#include <map>
#include <string>

namespace infini {

// TODO: delete line after change to Kernel
using MicroType = KernelType;

// using MicroAttrs =
//     std::tuple<PlatformType, MicroType, const int>;  // platfrom , type, ID

class Micro {
  /**
   * @brief Micro is smallest unit of codegen.
   *  It includes different types of memory access,
   *  operation codegen, and others
   */
 protected:
  MicroType micro_type;
  PlatformType platform;

 public:
  // Constructor
  Micro(){};
  Micro(const Micro &) = delete;
  // Destructor
  virtual ~Micro(){};

  /**
   * @brief Generate code. Subclasses rewrite this
   * function and call generations which belongs to
   * specific platform.
   */
  virtual std::string generatorCode(Cache &cache,
                                    std::string &result) const = 0;

  /** @brief Information print*/
  virtual void printInformation();
};

// class MicroRegistry {
//   /**
//    * @brief MicroRistry is a single Instancing class
//    * which contains different kinds of Micros
//    */
//  public:
//   using MicroRecord =
//       std::tuple<Micro *const, const std::string>;  // Micro* , name
//  private:
//   std::map<MicroAttrs, MicroRecord> micros;

//  public:
//   // Deconstruct
//   ~MicroRegistry() {
//     for (auto &[k, v] : micros) {
//       delete std::get<0>(v);
//     }
//   }
//   // Static function that makes MicroRegistry a class with single instance
//   static MicroRegistry &getInstance() {
//     static MicroRegistry instance;
//     return instance;
//   }
//   // Register Micro
//   bool RegisterMicro(const MicroAttrs &key, Micro *micro, std::string name) {
//     ASSERT(micros.find(key) == micros.end());  // micro not registered
//     micros.emplace(key, MicroRecord{micro, name});
//     return true;
//   }
//   // Get micro by MicroAttrs {PlatformType, microType}
//   Micro *getMicro(const MicroAttrs &microAttrs) const {
//     auto it = micros.find(microAttrs);
//     ASSERT(it != micros.end());
//     return std::get<0>(it->second);
//   }

//   const MicroRecord &getMicroRecord(const MicroAttrs &microAttrs) {
//     return micros.at(microAttrs);
//   }
// };

// #define REGISTER_MICRO(platform, microType, micro, name)              \
//   namespace infini {                                                  \
//   static const bool _register_micro_##__COUNTER__ =                   \
//       MicroRegistry::getInstance().registerMicro(                     \
//           MicroAttrs{platform, microType, __COUNTER__}, micro, name); \
//   }

}  // namespace infini
