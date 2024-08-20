#ifndef MICRO_H
#define MICRO_H
#include "core/cache.h"
#include "core/common.h"
#include "core/platform.h"
#include "core/tile.h"
#include "core/utils.h"
#include <functional>
#include <map>
#include <string>
#include <tuple>

namespace infini {

class Micro {
  protected:
    MicroType microType;
    Platform platform;

  public:
    // Constructor
    Micro(const Micro &) = delete;
    Micro() = default;
    Micro(MicroType microType_, Platform platform_)
        : microType(microType_), platform(platform_) {}
    // Destructor
    virtual ~Micro(){};

    virtual std::string code(Cache &cache, std::string &code,
                             int64_t indent = 0) = 0;
    static Micro *makeObj(); // dummpy functon
    virtual std::string info(bool print = true);
};

using MicroAttrs = std::tuple<OperatorType, Platform::underlying_t>;
using MicroConstructor = std::function<Micro *(
    const std::vector<Tile *> &inputs, const std::vector<Tile *> &outputs)>;

class MicroRegistry {
  private:
    std::map<MicroAttrs, MicroConstructor> microRecords;
    int numRecords = 0;

  public:
    ~MicroRegistry() = default;

    static MicroRegistry &getInstance() {
        static MicroRegistry instance;
        return instance;
    }
    bool registerMicro(const MicroAttrs &key, MicroConstructor constructor) {
        ASSERT(microRecords.find(key) == microRecords.end());
        microRecords.emplace(key, constructor);
        numRecords++;
        return true;
    }
    const MicroConstructor &getConstructor(const MicroAttrs &key) const {
        return microRecords.at(key);
    }
};

#define REGISTER_MICRO(optype, platform, constructor)                          \
    static const bool CAT(_register_micro_constructor_, __COUNTER__) =         \
        MicroRegistry::getInstance().registerMicro(                            \
            MicroAttrs{optype, platform}, constructor);

} // namespace infini

#endif
