#include "core/cache.h"
#include "core/utils.h"
#include <algorithm>
#include <cstdlib>

namespace infini {

Cache::Cache(int64_t total_nram, int64_t total_ldram, int64_t align_size,
             std::string name, MemoryDispatch dispatch) {
    name = name;
    cache_size = PAD_DOWN(total_nram, align_size);
    ldram_size = PAD_DOWN(total_ldram, align_size);
    cache_dispatch = dispatch;
    cache_align_size = align_size;
    clock = 0;

    // build cache block list
    Block *first_block = new Block(false, 0, total_nram, nullptr, nullptr, name,
                                   CacheType::CACHE, nullptr, -1);
    cache_head = new Block(true, 0, 0, first_block, nullptr, name,
                           CacheType::CACHE, nullptr, -1);
    cache_tail = new Block(true, total_nram, 0, nullptr, first_block, name,
                           CacheType::CACHE, nullptr, -1);
    first_block->prev = cache_head;
    first_block->next = cache_tail;
    free_cache_blocks = std::set<Block *, compareBlockSize>();
    free_cache_blocks.insert(first_block);

    // build ldram block list
    first_block = new Block(false, 0, total_ldram, nullptr, nullptr, name,
                            CacheType::LDRAM, nullptr, -1);
    ldram_head = new Block(true, 0, 0, first_block, nullptr, name,
                           CacheType::LDRAM, nullptr, -1);
    ldram_tail = new Block(true, total_nram, 0, nullptr, first_block, name,
                           CacheType::LDRAM, nullptr, -1);
    first_block->prev = ldram_head;
    first_block->next = ldram_tail;
    free_ldram_blocks = std::set<Block *, compareBlockSize>();
    free_ldram_blocks.insert(first_block);

    storedInLdram = std::unordered_set<CacheData, CacheDataHash>();
}

void Cache::clearCache() {
    clock = 0;
    Block *first_block = new Block(false, 0, cache_size, cache_tail, cache_head,
                                   name, CacheType::CACHE, nullptr, -1);
    cache_head->next = first_block;
    cache_tail->prev = first_block;
    free_cache_blocks = std::set<Block *, compareBlockSize>();
    free_cache_blocks.insert(cache_head);
}

void Cache::resetDispatch(MemoryDispatch dispatch) {
    this->clearCache();
    cache_dispatch = dispatch;
}

void Cache::initBlockCount(Block *block) {
    if (cache_dispatch == MemoryDispatch::FIFO) {
        block->data_count = clock;
        clock += 1;
    } else if (cache_dispatch == MemoryDispatch::LRU) {
        block->data_count = 0;
    } else if (cache_dispatch == MemoryDispatch::LFU) {
        block->data_count = 1;
    } else
        return;
}

void Cache::updateBlockCount(Block *block, bool match) {
    if (cache_dispatch == MemoryDispatch::FIFO && match) {
        block->data_count = clock;
        clock += 1;
    } else if (cache_dispatch == MemoryDispatch::LRU && !match) {
        block->data_count += 1;
    } else if (cache_dispatch == MemoryDispatch::LFU && match) {
        block->data_count += 1;
    } else
        return;
}

bool Cache::cacheReplaceable(Block *curr, Block *target) {
    if (cache_dispatch == MemoryDispatch::FIFO) {
        // FIFO: find the one with min timestamp
        return curr->data_count < target->data_count;
    } else if (cache_dispatch == MemoryDispatch::LRU) {
        // LRU: find the one with max time of not being used
        return curr->data_count > target->data_count;
    } else if (cache_dispatch == MemoryDispatch::LFU) {
        // LFU: find the one with min usage
        return curr->data_count < target->data_count;
    } else if (cache_dispatch == MemoryDispatch::RANDOM) {
        // RANDOM: randomly pick one, TODO
        float random_number = std::rand() % 1000 / 1000.0;
        return random_number < 0.5;
    } else
        return false;
}

void Cache::safeErase(Block *block) {
    if (block->cache_type == CacheType::CACHE) {
        if (!block->allocated) {
            auto it = free_cache_blocks.find(block);
            if (it != free_cache_blocks.end()) {
                free_cache_blocks.erase(it);
            }
        }
    } else {
        if (!block->allocated) {
            auto it = free_ldram_blocks.find(block);
            if (it != free_ldram_blocks.end()) {
                free_ldram_blocks.erase(it);
            }
        }
    }
}

std::vector<CacheData *> Cache::loadData2Block(CacheData *replacer_data,
                                               Block *replacee) {
    // initialize return data list
    std::vector<CacheData *> replacee_data_list;
    if (replacee->data != nullptr) {
        replacee_data_list.push_back(replacee->data);
    }
    int64_t data_size = PAD_UP(replacer_data->size, cache_align_size);
    int64_t remainder_size = replacee->block_size - data_size;
    if (remainder_size > 0) {
        // split to two blocks, and see if the 2nd block can be merged
        // to the following empty block
        // Note that we should change the params of replacee instead of creating
        // a new one bc its pointer will be used again
        if (replacee->next->allocated) {
            // naive split to two blocks
            replacee->block_size = data_size;
            replacee->data = replacer_data;
            Block *remainder = new Block(
                false, replacee->block_offset + data_size, remainder_size,
                replacee->next, replacee, replacee->cache_name,
                replacee->cache_type, nullptr, -1);
            replacee->next->prev = remainder;
            replacee->next = remainder;

            // update free block list
            safeErase(replacee);
            if (replacee->cache_type == CacheType::CACHE) {
                free_cache_blocks.insert(remainder);
            } else {
                free_ldram_blocks.insert(remainder);
            }
            replacee->allocated = true;
        } else {
            // there is a following empty block, so we need to do merging
            Block *next_empty_block = replacee->next;
            replacee->allocated = true;
            replacee->block_size = data_size;
            replacee->data = replacer_data;

            next_empty_block->block_offset = replacee->block_offset + data_size;
            next_empty_block->block_size += remainder_size;

            // in this case, replacee is supposed to be allocated
            // no need to update free block list
        }
    } else if (remainder_size == 0) {
        if (replacee->data != nullptr) {
            replacee_data_list.push_back(replacee->data);
        }
        // replace data in that block
        safeErase(replacee);
        replacee->allocated = true;
        replacee->data = replacer_data;
    } else if (remainder_size < 0) {
        // To take place of a series of blocks
        // Only happens in cache
        Block *ptr = replacee->next;
        int64_t replacee_total_size = replacee->block_size;
        while (ptr->next != nullptr && replacee_total_size < data_size) {
            replacee_total_size += ptr->block_size;
            if (ptr->allocated) {
                replacee_data_list.push_back(ptr->data);
            } else {
                safeErase(ptr);
            }
            ptr = ptr->next;
        }
        replacee->allocated = false;
        replacee->block_size = replacee_total_size;
        replacee->next = ptr;
        ptr->prev = replacee;
        replacee->data = nullptr;
        // in this case, replacee should be able to contain replacer_data
        // and since replacee->data is set to nullptr, return vec should be
        // empty
        loadData2Block(replacer_data, replacee);
        // TODO: delete memory of inbetween blocks
    }
    return replacee_data_list;
}

void Cache::freeBlock(Block *target) {
    target->allocated = false;
    target->data = nullptr;
    if (!target->next->allocated && target->prev->allocated) {
        target->block_size += target->next->block_size;
        // TODO: delete memory of target->next?
        // delete target->next;
        safeErase(target->next);
        target->next = target->next->next;
        target->next->prev = target;
    } else if (target->next->allocated && !target->prev->allocated) {
        target->block_size += target->prev->block_size;
        safeErase(target->prev);
        target->prev = target->prev->prev;
        target->prev->next = target;
    } else if (!target->next->allocated && !target->prev->allocated) {
        target->block_size += target->next->block_size;
        target->block_size += target->prev->block_size;
        safeErase(target->next);
        safeErase(target->prev);
        target->next = target->next->next;
        target->next->prev = target;
        target->prev = target->prev->prev;
        target->prev->next = target;
    }
    if (target->cache_type == CacheType::CACHE) {
        free_cache_blocks.insert(target);
    } else {
        free_ldram_blocks.insert(target);
    }
}

CacheHit Cache::loadData(CacheData *target_data) {
    int64_t size = target_data->size;
    if (size > cache_size) {
        LOG(ERROR) << "Cache size is less than data size.";
        return CacheHit(CacheHitLocation::ERROR, -1, -1, -1, -1);
    }

    bool match_cache = false;
    bool match_ldram = false;
    std::string result = "";
    Block *ptr = cache_head->next;
    std::string target_data_info = TO_STRING(*target_data);
    Block *first_allocated_block = nullptr;

    LOG(INFO) << "Looking for " + target_data_info + " in cache...";

    // Cache block to return
    Block *target_cache_block = nullptr;
    Block *ldram_from_block = nullptr;
    Block *ldram_to_block = nullptr;

    // 1. Find data in cache
    while (ptr->next != nullptr) {
        if (!ptr->allocated) {
            ptr = ptr->next;
            continue;
        }
        if (first_allocated_block == nullptr) {
            first_allocated_block = ptr;
        }
        if (ptr->data->equalsTo(*target_data)) {
            // OUTCOME 0: found in cache
            match_cache = true;
            std::string cache_info = TO_STRING(*ptr);
            LOG(INFO) << indentation(1) + "Cache hit, " + target_data_info +
                             " has been cached on " + cache_info;
            updateBlockCount(ptr, true);
            target_cache_block = ptr;
            break;
        } else {
            updateBlockCount(ptr, false);
        }
        ptr = ptr->next;
    }
    // 2. Not found in cache
    if (!match_cache) {
        LOG(INFO) << indentation(1) + "Could not find " + target_data_info +
                         " in cache.";
        int64_t data_size = PAD_UP(target_data->size, cache_align_size);
        // Find an empty cache block first
        Block *cmp = new Block(true, 0, data_size, nullptr, nullptr, "",
                               CacheType::CACHE, target_data, -1);
        auto cache_block_it = free_cache_blocks.lower_bound(cmp);
        if (cache_block_it != free_cache_blocks.end()) {
            // found an empty cache block
            target_cache_block = *cache_block_it;
            LOG(INFO) << indentation(2) + "Find an empty block " +
                             TO_STRING(*target_cache_block) + " to cache " +
                             target_data_info + ".";
        } else {
            // 3. Cache full, find one cache block to swap
            ptr = cache_head->next;
            target_cache_block = first_allocated_block;
            while (ptr->next != nullptr) {
                if (ptr->allocated &&
                    cacheReplaceable(ptr, target_cache_block) &&
                    cache_size - ptr->block_offset >= target_data->size) {
                    target_cache_block = ptr;
                }
                ptr = ptr->next;
            }
            if (target_cache_block == nullptr) {
                target_cache_block = cache_head->next;
            }
            // try best to minimize fragments
            if (!target_cache_block->prev->allocated) {
                target_cache_block = target_cache_block->prev;
            }
            LOG(INFO) << indentation(2) + "Cache Full. Find cache block " +
                             TO_STRING(*target_cache_block) + " to replace.";
        }
        // Then check ldram
        if (storedInLdram.count(*target_data) > 0) {
            Block *ldram_ptr = ldram_head->next;
            while (ldram_ptr->next != nullptr) {
                if (!ldram_ptr->allocated) {
                    ldram_ptr = ldram_ptr->next;
                    continue;
                }
                if (ldram_ptr->data->equalsTo(*target_data)) {
                    // OUTCOME 1: found in ldram
                    match_ldram = true;
                    ldram_from_block = ldram_ptr;
                    std::string cache_info = TO_STRING(*ldram_from_block);
                    LOG(INFO) << indentation(3) + "LDRAM hit, " +
                                     target_data_info +
                                     " has been previously stored in " +
                                     TO_STRING(*ldram_from_block);
                    break;
                }
                ldram_ptr = ldram_ptr->next;
            }
        }
        // No matter if found in ldram, a ldram_to_block is supposed to be
        // allocated for cache swapping. We also assume that ldram is large
        // enough so abort in case of any overflow
        Block *cmp_ldram = new Block(
            true, 0, std::max(target_cache_block->block_size, data_size),
            nullptr, nullptr, "", CacheType::LDRAM, target_cache_block->data,
            -1);
        auto ldram_block_it = free_ldram_blocks.lower_bound(cmp_ldram);
        if (ldram_block_it != free_ldram_blocks.end()) {
            ldram_to_block = *ldram_block_it;
            LOG(INFO) << indentation(3) + "Find LDRAM block " +
                             TO_STRING(*ldram_to_block) +
                             " to store the replaced data " +
                             TO_STRING(*target_cache_block) + " from cache.";
        } else {
            LOG(ERROR) << "LDRAM has no more space.";
            return CacheHit(CacheHitLocation::ERROR, -1, -1, -1, -1);
        }
    }
    // Now that we already know cache/ldram_from/ldram_to locations, we then
    // update the information in the block link list
    if (match_cache) {
        // OUTCOME 0: found in cache
        return CacheHit(CacheHitLocation::CACHE,
                        target_cache_block->block_offset, -1, -1, -1);
    } else {
        // OUTCOME 1/2: not found in cache
        // a. load target to cache
        auto replacee_data_list =
            loadData2Block(target_data, target_cache_block);
        initBlockCount(target_cache_block);
        // b. load replaced cache data to ldram
        Block *ldram_to_initial = ldram_to_block;
        if (!replacee_data_list.empty()) {
            for (auto replacee_data : replacee_data_list) {
                loadData2Block(replacee_data, ldram_to_block);
                storedInLdram.insert(*replacee_data);
                ldram_to_block = ldram_to_block->next;
            }
        }
        if (match_ldram) {
            // OUTCOME 1: found in ldram
            // c. free target from ldram
            freeBlock(ldram_from_block);
            storedInLdram.erase(*target_data);
            if (!replacee_data_list.empty()) {
                // OUTCOME 1.1: found in ldram, cache full
                return CacheHit(CacheHitLocation::LDRAM,
                                target_cache_block->block_offset,
                                ldram_from_block->block_offset,
                                ldram_to_initial->block_offset,
                                target_cache_block->block_size);
            } else {
                // OUTCOME 1.1: found in ldram, cache has space
                return CacheHit(CacheHitLocation::LDRAM,
                                target_cache_block->block_offset,
                                ldram_from_block->block_offset, -1, -1);
            }
        } else {
            // OUTCOME 2: not found at all
            if (!replacee_data_list.empty()) {
                // OUTCOME 2.1: not found at all, cache full
                return CacheHit(CacheHitLocation::NOT_FOUND,
                                target_cache_block->block_offset, -1,
                                ldram_to_initial->block_offset,
                                target_cache_block->block_size);
            } else {
                // OUTCOME 2.2: not found at all, cache has space
                return CacheHit(CacheHitLocation::NOT_FOUND,
                                target_cache_block->block_offset, -1, -1, -1);
            }
        }
    }
}

void Cache::printMemoryGraph(Block *head, int height = 16, int width = 64) {
    int64_t total =
        (head->cache_type == CacheType::CACHE) ? cache_size : ldram_size;
    std::string info_string =
        TO_STRING(head->cache_type) + " of " + head->cache_name;
    info_string += ", Size: " + std::to_string(total);
    LOG(INFO) << info_string;
    float unit_size = float(total) / float(height * width);

    std::string axis = std::string(8, ' ');
    int row_size = total / height;
    int interval_size = row_size / 4;
    int interval_len = width / 4;
    for (int i = 0; i < 4; i++) {
        axis += rpad(std::to_string(i * interval_size), interval_len, ' ');
    }
    LOG(INFO) << axis + std::to_string(row_size);

    int cur_row = 0;
    int cur_col = 0;
    Block *ptr = head->next;
    std::string characters = "#@$%&*=+ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string output = lpad(std::to_string(cur_row), 5, ' ') + " | ";
    while (ptr->next != nullptr) {
        int size = ptr->block_size;
        int num_units = size / unit_size;
        std::string status = "";
        if (ptr->allocated) {
            if (ptr->data->name.length() == 1) {
                status += ptr->data->name;
            } else {
                int index = CacheDataHash()(*(ptr->data)) % characters.length();
                status += characters[index];
            }
        } else {
            status += ".";
        }
        for (int i = 0; i < num_units; i++) {
            output += status;
            cur_col += 1;
            if (cur_col == width) {
                LOG(INFO) << output;
                cur_row += 1;
                output =
                    lpad(std::to_string(cur_row * row_size), 5, ' ') + " | ";
                cur_col = 0;
            }
        }
        ptr = ptr->next;
    }
}

void Cache::printInformation() {
    std::string info_string = "";
    info_string += "—— Cache ";
    info_string += "Name: ";
    info_string += name;
    info_string += ", ";
    info_string += "Cache Size: ";
    info_string += std::to_string(cache_size);
    info_string += ", ";
    info_string += "LDRAM Size: ";
    info_string += std::to_string(ldram_size);
    info_string += ", ";
    info_string += "Memory Dispatch: ";
    info_string += TO_STRING(cache_dispatch);
    LOG(INFO) << info_string;
    printMemoryGraph(cache_head);
    printBlocks(cache_head);
    printMemoryGraph(ldram_head);
    printBlocks(ldram_head);
}

void Cache::printBlocks(Block *head) {
    std::string info_string =
        std::string("  -- Data in " + TO_STRING(head->cache_type) + ": ");
    LOG(INFO) << info_string;
    Block *ptr = head->next;
    while (ptr->next != nullptr) {
        info_string = "         - Block ";
        info_string += "Offset: ";
        info_string += std::to_string(ptr->block_offset);
        info_string += "; ";
        info_string += "Size: ";
        info_string += std::to_string(ptr->block_size);
        info_string += "; ";
        if (!ptr->allocated) {
            info_string += "Empty";
        } else {
            info_string += "Data Name: ";
            info_string += ptr->data->name;
            info_string += "; ";
            info_string += "Data Size: ";
            info_string += std::to_string(ptr->data->size);
            info_string += "; ";
            info_string += "Data Count: ";
            info_string += std::to_string(ptr->data_count);
        }
        LOG(INFO) << info_string;
        ptr = ptr->next;
    }
}

} // namespace infini
