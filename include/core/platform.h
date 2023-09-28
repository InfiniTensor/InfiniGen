#pragma once
#include <string>
#include <vector>

namespace infini {

struct Platform {
  using underlying_t = uint16_t;

  enum : underlying_t { CUDA, BANG } type;

  constexpr Platform(decltype(type) t) : type(t) {}
  constexpr explicit Platform(underlying_t val) : type((decltype(type))val) {}
  constexpr underlying_t underlying() const { return type; }

  bool operator==(Platform others) const { return type == others.type; }
  bool operator!=(Platform others) const { return type != others.type; }

  // Print platform related code
  const std::string deviceFuncDecl(std::string name) const;
  const std::string globalFuncDecl(std::string name) const;
  const std::string taskId(int dim) const;
  const std::string taskId() const;
  const std::string taskDim(int dim) const;
  const std::string taskDim() const;
  const std::string regDecl(std::string datatype, std::string name) const;
  const std::string ldramDecl(std::string datatype, std::string name) const;
  const std::string shmemDecl(std::string datatype, std::string name) const;
  const std::string glmemDecl(std::string datatype, std::string name) const;
  const std::string queue() const;
  const std::string head() const;
  const std::string taskScaleDecl(int64_t num_tiles) const;
  const std::string syntacticSugar() const;

  const char *toString() const;
  bool isCUDA() const;
  bool isBANG() const;
};

}  // namespace infini
