#pragma once
#include "core/type.h"
#include <set>
#include <unordered_set>
#include <vector>

namespace infini {

struct CacheData {
    std::string name;
    int64_t offset;
    int64_t size;

    CacheData() = default;
    CacheData(std::string _name, int64_t _offset, int64_t _size)
        : name(_name), offset(_offset), size(_size) {}

    bool equalsTo(const CacheData &other) const {
        return name == other.name && offset == other.offset &&
               size == other.size;
    }

    bool operator==(const CacheData &other) const {
        return name == other.name && offset == other.offset &&
               size == other.size;
    }
};

struct CacheDataHash {
    size_t operator()(const CacheData &data) const {
        size_t name_hash = std::hash<std::string>{}(data.name);
        size_t offset_hash = std::hash<int64_t>{}(data.offset);
        size_t size_hash = std::hash<int64_t>{}(data.size);

        return name_hash ^ (offset_hash << 1) ^ (size_hash << 2);
    }
};

struct Block {
    bool allocated;
    int64_t block_offset;
    int64_t block_size;
    Block *next;
    Block *prev;
    std::string cache_name;
    CacheType cache_type;
    CacheData *data;
    int data_count;

    Block(bool _allocated, int64_t _block_offset, int64_t _block_size,
          Block *_next, Block *_prev, std::string _cache_name,
          CacheType _cache_type, CacheData *_data, int _data_count)
        : allocated(_allocated), block_offset(_block_offset),
          block_size(_block_size), next(_next), prev(_prev),
          cache_name(_cache_name), cache_type(_cache_type), data(_data),
          data_count(_data_count) {}
};

struct compareBlockSize {
    bool operator()(const Block *block1, const Block *block2) const {
        return block1->block_size < block2->block_size;
    }
};

struct CacheHit {
    CacheHitLocation location;
    int64_t cache_offset;
    int64_t ldram_from_offset;
    int64_t ldram_to_offset;
    int64_t replaced_data_size;

    CacheHit() = default;
    CacheHit(CacheHitLocation _location, int64_t _cache_offset,
             int64_t _ldram_from_offset, int64_t _ldram_to_offset,
             int64_t _replaced_data_size)
        : location(_location), cache_offset(_cache_offset),
          ldram_from_offset(_ldram_from_offset),
          ldram_to_offset(_ldram_to_offset),
          replaced_data_size(_replaced_data_size) {}
};

class Cache {
  public:
    // Self information
    int64_t cache_size;
    int64_t cache_align_size;
    int64_t ldram_size;
    int64_t clock; // for FIFO
    Block *cache_head;
    Block *cache_tail;
    Block *ldram_head;
    Block *ldram_tail;
    std::set<Block *, compareBlockSize> free_cache_blocks;
    std::set<Block *, compareBlockSize> free_ldram_blocks;
    std::unordered_set<CacheData, CacheDataHash> storedInLdram;
    std::unordered_set<CacheData, CacheDataHash> lockedData;
    MemoryDispatch cache_dispatch;
    std::string name;

  public:
    // Constructor
    Cache() = delete;
    Cache(int64_t total_nram, int64_t total_ldram, int64_t align_size,
          std::string name, MemoryDispatch dispatch);
    // Destructor
    ~Cache() = default;
    // Clear cache information
    void clearCache();
    // Reset cache dispatch algorithm
    void resetDispatch(MemoryDispatch dispatch);
    // Load data
    CacheHit load(CacheData *data);
    // Allocate memory for data
    CacheHit allocate(CacheData *data);
    // free data from cache
    int64_t free(CacheData *data);
    // Lock & unlock data
    void lock(std::vector<CacheData> data_list);
    void unlock(std::vector<CacheData> data_list);
    // Information
    void printInformation();
    void printBlocks(Block *head);

  private:
    void initBlockCount(Block *block);
    void updateBlockCount(Block *block, bool match);
    Block *cacheAlloc(CacheData *target_data, int indent);
    bool cacheReplaceable(Block *curr, Block *target);
    void safeErase(Block *block);
    std::vector<CacheData *> loadData2Block(CacheData *replacer_data,
                                            Block *replacee);
    void freeBlock(Block *target);
    void printMemoryGraph(Block *head, int height, int width);
};

} // namespace infini
