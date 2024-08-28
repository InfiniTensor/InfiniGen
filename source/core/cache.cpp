#include "core/cache.h"
#include "core/log.h"
#include "core/utils.h"
#include <limits>

namespace infini {

int64_t Cache::cacheCount = 0;

Block::Block(const int64_t &start, const int64_t &length)
    : blockStart(start), blockLength(length), blockPrecursor(nullptr),
      blockSuccessor(nullptr), blockTile(nullptr), blockTimestamp(-1) {}

Cache::Cache(const int64_t &size, const CachePolicy &policy,
             const std::string &name)
    : cacheSize(size),
      cacheName(name == "" ? "Cache_" + std::to_string(cacheCount) : name),
      cacheIndex(cacheCount++), cachePolicy(policy) {
    cacheFreeList = new Block(0, 0);
    cacheFreeList->blockSuccessor = cacheFreeList->blockPrecursor = nullptr;
    cacheFreeList->blockSuccessor = new Block(0, cacheSize);
    cacheFreeList->blockSuccessor->blockPrecursor = cacheFreeList;
    cacheOccupyList = new Block(0, 0);
    cacheOccupyList->blockSuccessor = cacheOccupyList->blockPrecursor = nullptr;
    cacheIsLocked = false;
    cacheClock = 0;
}

Cache::~Cache() {
    Block *freeCursor = cacheFreeList;
    while (freeCursor) {
        Block *temp = freeCursor;
        freeCursor = freeCursor->blockSuccessor;
        delete temp;
    }
    Block *occupyCursor = cacheOccupyList;
    while (occupyCursor) {
        Block *temp = occupyCursor;
        occupyCursor = occupyCursor->blockSuccessor;
        delete temp;
    }
}

std::string Cache::info(bool print) {
    std::stringstream out;
    out << BRIGHT_MAGENTA << HIGHLIGHT << "[CACHE] " << RESET;
    std::string prefix = out.str();
    out << cacheName << "(" << cacheSize << " Bytes, " << TO_STRING(cachePolicy)
        << ")" << std::endl;

    Block *occupyCursor = cacheOccupyList->blockSuccessor;
    while (occupyCursor) {
        std::string locked =
            (cacheLockedTiles.count(occupyCursor->blockTile) > 0) ? " (locked)"
                                                                  : "";
        out << prefix << YELLOW << "   Full" << RESET << ": ["
            << occupyCursor->blockStart << ", "
            << occupyCursor->blockStart + occupyCursor->blockLength
            << "](len: " << occupyCursor->blockLength
            << ", ts: " << occupyCursor->blockTimestamp << ") --> "
            << occupyCursor->blockTile->tileName << locked << std::endl;
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

void Cache::initBlockTimestamp(Block *block) {
    cacheClock += 1;
    if (cachePolicy == CachePolicy::FIFO) {
        block->blockTimestamp = cacheClock;
    } else if (cachePolicy == CachePolicy::LRU) {
        block->blockTimestamp = cacheClock;
    } else if (cachePolicy == CachePolicy::LFU) {
        block->blockTimestamp = 1;
    } else {
        block->blockTimestamp = 1;
    }
}

void Cache::updateBlockTimestamp(Block *block) {
    cacheClock += 1;
    if (cachePolicy == CachePolicy::FIFO) {
        return;
    } else if (cachePolicy == CachePolicy::LRU) {
        block->blockTimestamp = cacheClock;
    } else if (cachePolicy == CachePolicy::LFU) {
        block->blockTimestamp += 1;
    } else
        return;
}

void Cache::lock() { cacheIsLocked = true; }

void Cache::unlock() {
    cacheIsLocked = false;
    cacheLockedTiles.clear();
}

int64_t Cache::largestBlockSize(Block *head) {
    Block *cursor = head;
    int64_t largestSize = 0;

    while (cursor->blockSuccessor) {
        largestSize =
            std::max(cursor->blockSuccessor->blockLength, largestSize);
        cursor = cursor->blockSuccessor;
    }

    return largestSize;
}

Block *Cache::find(Tile *data) {
    auto it = cacheMap.find(data);
    if (it != cacheMap.end()) {
        return cacheMap[data];
    }
    return nullptr;
}

Block *Cache::load(Tile *data) {
    if (cacheIsLocked) {
        cacheLockedTiles.insert(data);
    }
    if (find(data)) {
        updateBlockTimestamp(cacheMap[data]);
        return cacheMap[data];
    } else {
        return allocate(data);
    }
}

Block *Cache::allocate(Tile *data) {
    if (cacheIsLocked) {
        cacheLockedTiles.insert(data);
    }
    auto it = cacheMap.find(data);
    if (it != cacheMap.end()) {
        return cacheMap[data];
    }
    int64_t size = data->getSizeInBytes();

    // Cache replacement
    if (largestBlockSize(cacheFreeList) < size) {
        auto swapResult = swapOut(size);
        if (!swapResult) {
            LOG(ERROR) << "No space for load.";
            return nullptr;
        }
    }

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
            initBlockTimestamp(result);
            return result;
        } else if (freeCursor->blockSuccessor->blockLength == size) {
            Block *result =
                new Block(freeCursor->blockSuccessor->blockStart, size);
            result->blockTile = data;
            if (freeCursor->blockSuccessor->blockSuccessor) {
                Block *temp = freeCursor->blockSuccessor;
                temp->blockSuccessor->blockPrecursor = freeCursor;
                freeCursor->blockSuccessor = temp->blockSuccessor;
                delete temp;
            } else {
                Block *temp = freeCursor->blockSuccessor;
                delete temp;
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
            initBlockTimestamp(result);
            return result;
        }
        freeCursor = freeCursor->blockSuccessor;
    }
    // Should not get here
    LOG(ERROR) << "No space for load.";
}

bool Cache::swapOut(int64_t size) {
    while (largestBlockSize(cacheFreeList) < size) {
        Block *occupyCursor = cacheOccupyList;
        Block *replaceCandidate = nullptr;
        int64_t minTimestamp = std::numeric_limits<int64_t>::max();

        while (occupyCursor->blockSuccessor) {
            auto block = occupyCursor->blockSuccessor;
            if (cacheLockedTiles.count(block->blockTile) == 0 &&
                block->blockTimestamp > 0 &&
                block->blockTimestamp < minTimestamp) {
                replaceCandidate = block;
                minTimestamp = block->blockTimestamp;
            }
            occupyCursor = occupyCursor->blockSuccessor;
        }

        if (!replaceCandidate) {
            return false;
        }

        this->free(replaceCandidate);
    }
    return true;
}

void Cache::free(Tile *data) {
    if (cacheLockedTiles.count(data) > 0) {
        LOG(ERROR) << "Data in use. Cannot Free.";
        return;
    }
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
            // Remove block from cacheOccupyList
            if (block->blockSuccessor) {
                occupyCursor->blockSuccessor = block->blockSuccessor;
                block->blockSuccessor->blockPrecursor = occupyCursor;
            } else {
                occupyCursor->blockSuccessor = nullptr;
            }
            // Merge block into cacheFreeList
            Block *freeCursor = cacheFreeList;
            Block *prev = nullptr, *next = nullptr;
            while (freeCursor->blockSuccessor) {
                if (block->blockStart + block->blockLength ==
                    freeCursor->blockSuccessor->blockStart) {
                    next = freeCursor->blockSuccessor;
                } else if (freeCursor->blockSuccessor->blockStart +
                               freeCursor->blockSuccessor->blockLength ==
                           block->blockStart) {
                    prev = freeCursor->blockSuccessor;
                }
                freeCursor = freeCursor->blockSuccessor;
            }

            if (prev && next) {
                prev->blockLength += block->blockLength + next->blockLength;
                next->blockPrecursor->blockSuccessor = next->blockSuccessor;
                if (next->blockSuccessor) {
                    next->blockSuccessor->blockPrecursor = next->blockPrecursor;
                }
                delete block;
                delete next;
                return;
            } else if (prev) {
                prev->blockLength += block->blockLength;
                delete block;
                return;
            } else if (next) {
                next->blockStart = block->blockStart;
                next->blockLength += block->blockLength;
                delete block;
                return;
            } else {
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
