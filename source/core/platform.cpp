#include "core/platform.h"

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

const std::string Platform::taskIdx(int dim) const {
  std::vector<std::string> dim_map = {".x", ".y", ".z"};
  switch (type) {
    CASE(CUDA, "blockIdx" + dim_map[dim]);
    CASE(BANG, "taskIdx" + dim_map[dim]);
    default:
      return "";
  }
}

const std::string Platform::taskIdx() const {
  switch (type) {
    CASE(CUDA, "blockIdx");
    CASE(BANG, "taskIdx");
    default:
      return "";
  }
}

const std::string Platform::taskDim(int dim) const {
  std::vector<std::string> dim_map = {".x", ".y", ".z"};
  switch (type) {
    CASE(CUDA, "blockDim" + dim_map[dim]);
    CASE(BANG, "taskDim" + dim_map[dim]);
    default:
      return "";
  }
}

const std::string Platform::taskDim() const {
  switch (type) {
    CASE(CUDA, "blockDim");
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

const char* Platform::toString() const {
  switch (type) {
    CASE(CUDA, "CUDA");
    CASE(BANG, "BANG");
    default:
      return "Unknown";
  }
}

bool Platform::isCUDA() const { return type == Platform::CUDA; }

bool Platform::isBANG() const { return type == Platform::BANG; }

}  // namespace infini
