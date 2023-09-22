#pragma once
#include "core/api.h"
#include <string>
#include <tuple>

namespace infini {

struct Platform {
  using underlying_t = uint16_t;

  enum : underlying_t { CUDA, BANG } type;

  constexpr Platform(decltype(type) t) : type(t) {}
  constexpr explicit Platform(underlying_t val)
      : type((decltype(type))val) {}
  constexpr underlying_t underlying() const { return type; }

  bool operator==(Platform others) const { return type == others.type; }
  bool operator!=(Platform others) const { return type != others.type; }

  // Print platform related code
  const std::string deviceFuncDecl() const;
  const std::string globalFuncDecl() const;
  const std::string taskIdxDecl(int dim=0) const;
  const std::string taskDimDecl(int dim=0) const;
  const std::string regDecl() const;
  const std::string ldramDecl() const;
  const std::string shmemDecl() const;
  const std::string glmemDecl() const;

  const char *toString() const;
  bool isCUDA() const;
  bool isBANG() const;
};

}  // namespace infini