#ifndef GENERATOR_H
#define GENERATOR_H
#include "core/cache.h"
#include "core/common.h"
#include "core/graph.h"
#include "core/micro.h"
#include "core/platform.h"

namespace infini {

class Generator {
  protected:
    Graph *graph;
    Cache cache;
    Shape pattern;
    MicroRegistry registry;
    std::vector<std::pair<std::string, std::string>> params;

  public:
    std::vector<Micro *> microList;
    Platform platform;

  public:
    Generator() = delete;
    Generator(Platform platform, Graph *graph, Shape pattern = {},
              int64_t cacheSize = 40960);
    ~Generator() = default;

    std::string generateCode();

  private:
    Shape getProperTileShape();
};

} // namespace infini

#endif
