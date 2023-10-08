#include "core/platform.h"
#include "core/utils.h"

namespace infini {

#define CASE(TYPE, STR) \
  case Platform::TYPE:  \
    return STR

const std::string Platform::deviceFuncDecl(std::string name) const {
  switch (type) {
    CASE(CUDA, "__device__ void " + name);
    CASE(BANG, "__mlu_func__ void " + name);
    default:
      return "";
  }
}

const std::string Platform::globalFuncDecl(std::string name) const {
  switch (type) {
    CASE(CUDA, "__global__ void " + name);
    CASE(BANG, "__mlu_entry__ void " + name);
    default:
      return "";
  }
}

const std::string Platform::taskId(int dim) const {
  // Coordinates of block/task on device
  std::vector<std::string> dim_map_cuda = {".x", ".y", ".z"};
  std::vector<std::string> dim_map_bang = {"X", "Y", "Z"};
  switch (type) {
    CASE(CUDA, "blockIdx" + dim_map_cuda[dim]);
    CASE(BANG, "taskId" + dim_map_bang[dim]);
    default:
      return "";
  }
}

const std::string Platform::taskId() const {
  // Linear ID of block/task on device
  switch (type) {
    CASE(CUDA,
         "(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * "
         "gridDim.y)");
    CASE(BANG, "taskId");
    default:
      return "";
  }
}

const std::string Platform::taskDim(int dim) const {
  // Number of blocks/tasks on each dimension
  std::vector<std::string> dim_map_cuda = {".x", ".y", ".z"};
  std::vector<std::string> dim_map_bang = {"X", "Y", "Z"};
  switch (type) {
    CASE(CUDA, "gridDim" + dim_map_cuda[dim]);
    CASE(BANG, "taskDim" + dim_map_bang[dim]);
    default:
      return "";
  }
}

const std::string Platform::taskDim() const {
  // Total number of blocks/tasks
  switch (type) {
    CASE(CUDA, "(gridDim.x * gridDim.y * gridDim.z)");
    CASE(BANG, "taskDim");
    default:
      return "";
  }
}

const std::string Platform::regDecl(std::string datatype,
                                    std::string name) const {
  switch (type) {
    CASE(CUDA, datatype + " " + name);
    CASE(BANG, "__nram__ " + datatype + " " + name);
    default:
      return "";
  }
}

const std::string Platform::ldramDecl(std::string datatype,
                                      std::string name) const {
  switch (type) {
    CASE(CUDA, datatype + " " + name);  // 不确定是不是这个
    CASE(BANG, "__ldram__ " + datatype + " " + name);
    default:
      return "";
  }
}

const std::string Platform::shmemDecl(std::string datatype,
                                      std::string name) const {
  switch (type) {
    CASE(CUDA, "__shared__ " + datatype + " " + name);
    CASE(BANG, "__mlu_shared__ " + datatype + " " + name);
    default:
      return "";
  }
}

const std::string Platform::glmemDecl(std::string datatype,
                                      std::string name) const {
  switch (type) {
    CASE(CUDA, "__device__ " + datatype + " " + name);
    CASE(BANG, "__mlu_device__ " + datatype + " " + name);
    default:
      return "";
  }
}

const std::string Platform::queue() const {
  switch (type) {
    CASE(CUDA, "cudaStream_t");
    CASE(BANG, "cnrtQueue_t");
    default:
      return "";
  }
}

const std::string Platform::head() const {
  switch (type) {
    CASE(CUDA, "#include <cuda.h>");
    CASE(BANG, "#include <bang.h>");
    default:
      return "";
  }
}

const char* Platform::toString() const {
  switch (type) {
    CASE(CUDA, "CUDA");
    CASE(BANG, "BANG");
    default:
      return "Unknown";
  }
}

const std::string Platform::taskScaleDecl(TileTensor tiles) const {
  switch (type) {
    CASE(CUDA,
         "int numBlocks = " + std::to_string(tiles.numNeatTiles()) +
             ", threadsPerBlock = " +
             std::to_string(VECTOR_PRODUCT(tiles.tiles[0].tile_dimension)) +
             ";");

    CASE(BANG, "cnrtDim3_t dim = {" +
                   std::to_string(PAD_UP(tiles.numNeatTiles(), 4)) +
                   ", 1, 1};");
    default:
      return "";
  }
}

const std::string Platform::syntacticSugar() const {
  switch (type) {
    CASE(CUDA, "<<<numBlocks, threadsPerBlock, 0, queue>>>");
    CASE(BANG, "<<<dim, CNRT_FUNC_TYPE_UNION1, queue>>>");
    default:
      return "";
  }
}

const std::string Platform::workingCoreCond(TileTensor tiles) const {
  switch (type) {
    CASE(CUDA, "");
    CASE(BANG, "if (taskId >= " + std::to_string(tiles.numNeatTiles()) +
                   ") { return; }");
    default:
      return "";
  }
}

const std::string Platform::cacheDecl(std::string name, int64_t cache_size,
                                      std::string datatype) const {
  switch (type) {
    CASE(CUDA, "char " + name + "[" + std::to_string(cache_size) + "];");
    CASE(BANG,
         "__nram__ char " + name + "[" + std::to_string(cache_size) + "];");
    default:
      return "";
  }
}

const std::string Platform::ldramDecl(std::string name,
                                      int64_t ldram_size) const {
  switch (type) {
    CASE(CUDA, "");
    CASE(BANG, "__ldram__ char " + name + "_ldram[" +
                   std::to_string(ldram_size) + "];");
    default:
      return "";
  }
}

bool Platform::isCUDA() const { return type == Platform::CUDA; }

bool Platform::isBANG() const { return type == Platform::BANG; }

}  // namespace infini
