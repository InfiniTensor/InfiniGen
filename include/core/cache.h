#ifndef CACHE_H
#define CACHE_H
#include "core/common.h"
#include "core/tile.h"
#include <unordered_map>

namespace infini {

class Tile;

class Block {
  public:
    int64_t blockStart;
    int64_t blockLength;
    Block *blockPrecursor;
    Block *blockSuccessor;
    Tile *blockTile;

  public:
    Block(const int64_t &start, const int64_t &length);
    ~Block() = default;
};

class Cache {
  private:
    static int64_t cacheCount;

  public:
    std::string cacheName;
    const int64_t cacheIndex;
    const int64_t cacheSize;
    Block *cacheFreeList;
    Block *cacheOccupyList;
    std::unordered_map<Tile *, Block *> cacheMap;

  public:
    Cache(const int64_t &size, const std::string &name = "");
    ~Cache();

    std::string info(bool print = true);

    Block *find(Tile *data);
    Block *load(Tile *data);
    Block *allocate(Tile *data);
    void free(Tile *data);
    void free(Block *block);
};

} // namespace infini
#endif
