#ifndef GENERATOR_H
#define GENERATOR_H
#include "core/cache.h"
#include "core/common.h"
#include "core/graph.h"
#include "core/micro.h"
#include "core/platform.h"

namespace infini {

struct Code {
    // Reusable code snippets for convenience
    std::string dataType;
    std::string params;
    std::string args;
};

class Generator {
  protected:
    Graph *graph;
    Cache cache;
    Shape pattern;
    MicroRegistry registry;
    Code code;

  public:
    std::vector<Micro *> microList;
    Platform platform;

  public:
    Generator() = delete;
    Generator(Platform platform, Graph *graph, Shape pattern = {},
              int64_t cacheSize = 40960);
    ~Generator() = default;

    std::string generateHeaderFile(const std::string &filepath = "",
                                   const int64_t &indent = 0);
    std::string generateSourceFile(const std::string &filepath = "",
                                   const int64_t &indent = 0);

#ifdef DEBUG_MODE
    std::string generateTestScript(const std::string &templateFilepath,
                                   const std::string &formula,
                                   const std::string &filepath = "");
#endif

  private:
    Shape getProperTileShape();
};

} // namespace infini

#endif
