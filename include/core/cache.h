#ifndef CACHE_H
#define CACHE_H
#include "core/common.h"
#include "core/tile.h"
#include <unordered_map>
#include <unordered_set>

namespace infini {

class Tile;

class Block {
  public:
    int64_t blockStart;
    int64_t blockLength;
    Block *blockPrecursor;
    Block *blockSuccessor;
    Tile *blockTile;
    int64_t blockTimestamp;

  public:
    Block(const int64_t &start, const int64_t &length);
    ~Block() = default;
};

class Cache {
  private:
    static int64_t cacheCount;

    Block *cacheFreeList;
    Block *cacheOccupyList;
    std::unordered_map<Tile *, Block *> cacheMap;
    std::unordered_set<Tile *> cacheLockedTiles;

    int64_t cacheClock;
    bool cacheIsLocked;
    CachePolicy cachePolicy;

  public:
    std::string cacheName;
    const int64_t cacheIndex;
    const int64_t cacheSize;

  public:
    Cache(const int64_t &size, const CachePolicy &policy = CachePolicy::LRU,
          const std::string &name = "");
    ~Cache();

    std::string info(bool print = true);

    Block *find(Tile *data);
    Block *load(Tile *data);
    Block *allocate(Tile *data);
    void free(Tile *data);

    void lock();
    void unlock();

  private:
    void free(Block *block);
    void initBlockTimestamp(Block *block);
    void updateBlockTimestamp(Block *block);
    bool swapOut(int64_t size);
    int64_t largestBlockSize(Block *head);
};

} // namespace infini
#endif
