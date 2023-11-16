#pragma once
#include <fmt/core.h>

#include "core/graph.h"

namespace infini {

class GemmGraph : public Graph {
 public:
  TileTensor tiles;
  std::vector<size_t> thread_block_size;
  std::vector<size_t> warp_size;
  std::vector<size_t> thread_size;

 public:
  // Constructor
  GemmGraph(std::vector<Node *> operators_list = {},
            std::vector<Data *> inputs_list = {},
            std::vector<Data *> outputs_list = {}, std::string name_value = "");
  // Destructor
  ~GemmGraph() = default;
  // Generator
  std::string generatorHead(int64_t indent) override;
  std::string generatorTask(int64_t indent) override;
  std::string generatorHost(int64_t indent) override;
  std::string generatorCode(int64_t indent) override;
  std::string generatorHeadFile(int64_t indent) override;
  std::string generatorSourceFile(int64_t indent) override;
  void applyPlatform(Platform platform) override;
  void split(std::vector<size_t> thread_block_size,
             std::vector<size_t> warp_size, std::vector<size_t> thread_size);
};

}  // namespace infini