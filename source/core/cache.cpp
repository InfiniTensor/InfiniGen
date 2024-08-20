#include "core/cache.h"
#include "core/log.h"
#include "core/utils.h"

namespace infini {

int64_t Cache::cacheCount = 0;

Block::Block(const int64_t &start, const int64_t &length)
    : blockStart(start), blockLength(length), blockPrecursor(nullptr),
      blockSuccessor(nullptr), blockTile(nullptr) {}

Cache::Cache(const int64_t &size, const std::string &name)
    : cacheSize(size),
      cacheName(name == "" ? "Cache_" + std::to_string(cacheCount) : name),
      cacheIndex(cacheCount++) {
    cacheFreeList = (Block *)malloc(sizeof(Block));
    cacheFreeList->blockSuccessor = cacheFreeList->blockPrecursor = nullptr;
    cacheFreeList->blockSuccessor = new Block(0, cacheSize);
    cacheFreeList->blockSuccessor->blockPrecursor = cacheFreeList;
    cacheOccupyList = (Block *)malloc(sizeof(Block));
    cacheOccupyList->blockSuccessor = cacheOccupyList->blockPrecursor = nullptr;
}

Cache::~Cache() {
    Block *freeCursor = cacheFreeList;
    while (freeCursor) {
        Block *temp = freeCursor;
        free(temp);
        freeCursor = freeCursor->blockSuccessor;
    }
    Block *occupyCursor = cacheOccupyList;
    while (occupyCursor) {
        Block *temp = occupyCursor;
        free(temp);
        occupyCursor = occupyCursor->blockSuccessor;
    }
}

std::string Cache::info(bool print) {
    std::stringstream out;
    out << BRIGHT_MAGENTA << HIGHLIGHT << "[CACHE] " << RESET;
    std::string prefix = out.str();
    out << cacheName << "(" << cacheSize << " Bytes)" << std::endl;

    Block *occupyCursor = cacheOccupyList->blockSuccessor;
    while (occupyCursor) {
        out << prefix << YELLOW << "   Full" << RESET << ": ["
            << occupyCursor->blockStart << ", "
            << occupyCursor->blockStart + occupyCursor->blockLength
            << "](len: " << occupyCursor->blockLength << ") --> "
            << occupyCursor->blockTile->tileName << std::endl;
        occupyCursor = occupyCursor->blockSuccessor;
    }

    Block *freeCursor = cacheFreeList->blockSuccessor;
    while (freeCursor) {
        out << prefix << GREEN << "   Free" << RESET << ": ["
            << freeCursor->blockStart << ", "
            << freeCursor->blockStart + freeCursor->blockLength
            << "](len: " << freeCursor->blockLength << ")" << std::endl;
        freeCursor = freeCursor->blockSuccessor;
    }

    if (print) {
        std::istringstream iss(out.str());
        std::string line;
        while (std::getline(iss, line)) {
            LOG(INFO) << line;
        }
    }
    return out.str();
}

Block *Cache::find(Tile *data) {
    auto it = cacheMap.find(data);
    if (it != cacheMap.end()) {
        return cacheMap[data];
    }
    return nullptr;
}

Block *Cache::load(Tile *data) {
    if (find(data)) {
        return cacheMap[data];
    } else {
        return allocate(data);
    }
}

Block *Cache::allocate(Tile *data) {
    auto it = cacheMap.find(data);
    if (it != cacheMap.end()) {
        return cacheMap[data];
    }
    int64_t size = data->getSizeInBytes();
    Block *freeCursor = cacheFreeList;
    while (freeCursor->blockSuccessor) {
        if (freeCursor->blockSuccessor->blockLength > size) {
            Block *result =
                new Block(freeCursor->blockSuccessor->blockStart, size);
            result->blockTile = data;
            freeCursor->blockSuccessor->blockStart += size;
            freeCursor->blockSuccessor->blockLength -= size;
            if (cacheOccupyList->blockSuccessor) {
                result->blockSuccessor = cacheOccupyList->blockSuccessor;
                result->blockPrecursor = cacheOccupyList;
                cacheOccupyList->blockSuccessor->blockPrecursor = result;
                cacheOccupyList->blockSuccessor = result;
            } else {
                cacheOccupyList->blockSuccessor = result;
                result->blockPrecursor = cacheOccupyList;
            }
            cacheMap[data] = result;
            return result;
        } else if (freeCursor->blockSuccessor->blockLength == size) {
            Block *result =
                new Block(freeCursor->blockSuccessor->blockStart, size);
            result->blockTile = data;
            if (freeCursor->blockSuccessor->blockSuccessor) {
                Block *temp = freeCursor->blockSuccessor;
                temp->blockSuccessor->blockPrecursor = freeCursor;
                freeCursor->blockSuccessor = temp->blockSuccessor;
                free(temp);
            } else {
                Block *temp = freeCursor->blockSuccessor;
                free(temp);
                freeCursor->blockSuccessor = nullptr;
            }
            if (cacheOccupyList->blockSuccessor) {
                result->blockSuccessor = cacheOccupyList->blockSuccessor;
                result->blockPrecursor = cacheOccupyList;
                cacheOccupyList->blockSuccessor->blockPrecursor = result;
                cacheOccupyList->blockSuccessor = result;
            } else {
                cacheOccupyList->blockSuccessor = result;
                result->blockPrecursor = cacheOccupyList;
            }
            cacheMap[data] = result;
            return result;
        }
        freeCursor = freeCursor->blockSuccessor;
    }
    // TODO: Cache swap strategy
    LOG(ERROR) << "No space for load.";
}

void Cache::free(Tile *data) {
    auto it = cacheMap.find(data);
    if (it != cacheMap.end()) {
        this->free(cacheMap[data]);
    } else {
        LOG(ERROR) << "Data not found.";
    }
}

void Cache::free(Block *block) {
    for (auto it = cacheMap.begin(); it != cacheMap.end(); ++it) {
        if (it->second == block) {
            cacheMap.erase(it);
            break;
        }
    }
    Block *occupyCursor = cacheOccupyList;
    while (occupyCursor->blockSuccessor) {
        if (occupyCursor->blockSuccessor == block) {
            if (block->blockSuccessor) {
                occupyCursor->blockSuccessor = block->blockSuccessor;
                block->blockSuccessor->blockPrecursor = occupyCursor;
                Block *freeCursor = cacheFreeList;
                while (freeCursor->blockSuccessor) {
                    if (block->blockStart + block->blockLength ==
                        freeCursor->blockSuccessor->blockStart) {
                        freeCursor->blockSuccessor->blockStart =
                            block->blockStart;
                        freeCursor->blockSuccessor->blockLength +=
                            block->blockLength;
                        free(block);
                        return;
                    } else if (freeCursor->blockSuccessor->blockStart +
                                   freeCursor->blockSuccessor->blockLength ==
                               block->blockStart) {
                        freeCursor->blockSuccessor->blockLength +=
                            block->blockLength;
                        free(block);
                        return;
                    }
                    freeCursor = freeCursor->blockSuccessor;
                }
                freeCursor->blockSuccessor = block;
                block->blockPrecursor = freeCursor;
                block->blockSuccessor = nullptr;
                return;
            } else {
                occupyCursor->blockSuccessor = nullptr;
                Block *freeCursor = cacheFreeList;
                while (freeCursor->blockSuccessor) {
                    if (block->blockStart + block->blockLength ==
                        freeCursor->blockSuccessor->blockStart) {
                        freeCursor->blockSuccessor->blockStart =
                            block->blockStart;
                        freeCursor->blockSuccessor->blockLength +=
                            block->blockLength;
                        free(block);
                        return;
                    } else if (freeCursor->blockSuccessor->blockStart +
                                   freeCursor->blockSuccessor->blockLength ==
                               block->blockStart) {
                        freeCursor->blockSuccessor->blockLength +=
                            block->blockLength;
                        free(block);
                        return;
                    }
                    freeCursor = freeCursor->blockSuccessor;
                }
                freeCursor->blockSuccessor = block;
                block->blockPrecursor = freeCursor;
                block->blockSuccessor = nullptr;
                return;
            }
        }
        occupyCursor = occupyCursor->blockSuccessor;
    }
}

} // namespace infini
