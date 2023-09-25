#include "core/platform.h"

namespace infini {

#define CASE(TYPE, STR)    \
  case Platform::TYPE: \
    return STR

const std::string Platform::deviceFuncDecl() const {
  switch (type) {
    CASE(CUDA, "__device__ void");
    CASE(BANG, "__mlu_func__ void");
    default:
      return "";
  }
}

const std::string Platform::globalFuncDecl() const {
  switch (type) {
    CASE(CUDA, "__global__ void");
    CASE(BANG, "__mlu_entry__ void");
    default:
      return "";
  }
}

const std::string Platform::taskIdxDecl(int dim) const {
  std::vector<std::string> dim_map = {".x", ".y", ".z"};
  switch (type) {
    CASE(CUDA, "blockIdx" + dim_map[dim]);
    CASE(BANG, "taskIdx" + dim_map[dim]);  
    default:
      return "";
  }
}

const std::string Platform::taskIdxDecl() const {
   switch (type) {
    CASE(CUDA, "blockIdx");
    CASE(BANG, "taskIdx");  
    default:
      return "";
  } 
}


const std::string Platform::taskDimDecl(int dim) const {
  std::vector<std::string> dim_map = {".x", ".y", ".z"};
  switch (type) {
    CASE(CUDA, "blockDim" + dim_map[dim]);
    CASE(BANG, "taskDim" + dim_map[dim]);
    default:
      return "";
  }
}

const std::string Platform::taskDimDecl() const {
  switch (type) {
    CASE(CUDA, "blockDim");
    CASE(BANG, "taskDim");
    default:
      return "";
  }
}

const std::string Platform::regDecl() const {
  switch (type) {
    CASE(CUDA, "");
    CASE(BANG, "__nram__");
    default:
      return "";
  }
}

const std::string Platform::ldramDecl() const {
  switch (type) {
    CASE(CUDA, ""); //不确定是不是这个
    CASE(BANG, "__ldram__");
    default:
      return "";
  }
}

const std::string Platform::shmemDecl() const {
  switch (type) {
    CASE(CUDA, "__shared__");
    CASE(BANG, "__mlu_shared__");
    default:
      return "";
  }
}

const std::string Platform::glmemDecl() const {
   switch (type) {
    CASE(CUDA, "__device__");
    CASE(BANG, "__mlu_device__");
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
