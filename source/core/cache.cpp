#include "core/cache.h"
#include "core/utils.h"
#include <algorithm>
#include <cstdlib>

namespace infini {

CacheData::CacheData(std::string _name, int64_t _offset, int64_t _size)
    : name(_name), offset(_offset), size(_size) {}

bool CacheData::operator==(const CacheData &other) const {
  return name == other.name && offset == other.offset && size == other.size;
}

size_t CacheDataHash::operator()(const CacheData &data) const {
  size_t name_hash = std::hash<std::string>{}(data.name);
  size_t offset_hash = std::hash<int64_t>{}(data.offset);
  size_t size_hash = std::hash<int64_t>{}(data.size);

  return name_hash ^ (offset_hash << 1) ^ (size_hash << 2);
}

Block::Block(bool _allocated, int64_t _block_offset, int64_t _block_size,
             Block *_next, Block *_prev, std::string _cache_name,
             CacheType _cache_type, CacheData *_data, int _data_count)
    : allocated(_allocated),
      block_offset(_block_offset),
      block_size(_block_size),
      next(_next),
      prev(_prev),
      cache_name(_cache_name),
      cache_type(_cache_type),
      data(_data),
      data_count(_data_count) {}

bool Block::operator==(const Block &block) const {
  return cache_type == block.cache_type && cache_name == block.cache_name &&
         block_offset == block.block_offset && block_size == block.block_size;
}

bool CompareBlockSize::operator()(const Block *block1,
                                  const Block *block2) const {
  // Since std::set uses key to sort as well as distinguishing among elements,
  // We create a new key for comparison that does not affect the original order
  // of block size and includes offset information so the key is able to be an
  // identifier for blocks (No need to include cache name and type, since we
  // assume that one std::set only contains blocks in the same location).
  //
  // Key: size.offset (size as integer part, offset as fraction part)
  auto blockHash = [](Block block) -> double {
    double offset = block.block_offset / 1.0;
    while (offset > 1.0) {
      offset /= 10.0;
    }
    return block.block_size / 1.0 + offset;
  };

  return blockHash(*block1) < blockHash(*block2);
}

CacheHit::CacheHit(
    CacheHitLocation _location, int64_t _cache_offset = -1,
    int64_t _ldram_from_offset = -1,
    std::vector<int64_t> _ldram_to_offset = std::vector<int64_t>(),
    std::vector<int64_t> _replaced_data_size = std::vector<int64_t>())
    : location(_location),
      cache_offset(_cache_offset),
      ldram_from_offset(_ldram_from_offset),
      ldram_to_offset(_ldram_to_offset),
      replaced_data_size(_replaced_data_size) {}

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
  free_cache_blocks = std::set<Block *, CompareBlockSize>();
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
  free_ldram_blocks = std::set<Block *, CompareBlockSize>();
  free_ldram_blocks.insert(first_block);

  storedInLdram = std::unordered_set<CacheData, CacheDataHash>();
  lockedData = std::unordered_set<CacheData, CacheDataHash>();
}

Cache::~Cache() {
  std::vector<Block *> blocks_to_delete;
  Block *ptr = cache_head->next;
  while (ptr->next != nullptr) {
    blocks_to_delete.push_back(ptr);
    ptr = ptr->next;
  }
  ptr = ldram_head->next;
  while (ptr->next != nullptr) {
    blocks_to_delete.push_back(ptr);
    ptr = ptr->next;
  }
  for (auto block : blocks_to_delete) {
    delete block;
  }
  delete cache_head;
  delete cache_tail;
  delete ldram_head;
  delete ldram_tail;
}

void Cache::clearCache() {
  clock = 0;
  std::vector<Block *> blocks_to_delete;
  Block *ptr = cache_head->next;
  while (ptr->next != nullptr) {
    blocks_to_delete.push_back(ptr);
    ptr = ptr->next;
  }
  for (auto block : blocks_to_delete) {
    delete block;
  }
  Block *first_block = new Block(false, 0, cache_size, cache_tail, cache_head,
                                 name, CacheType::CACHE, nullptr, -1);
  cache_head->next = first_block;
  cache_tail->prev = first_block;
  free_cache_blocks = std::set<Block *, CompareBlockSize>();
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
  if (lockedData.count(*(target->data)) > 0) {
    // should not replace any block with locked data
    return false;
  }
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

void Cache::lock(std::vector<CacheData> data_list) {
  for (auto data : data_list) {
    lockedData.insert(data);
  }
}

void Cache::unlock(std::vector<CacheData> data_list) {
  for (auto data : data_list) {
    lockedData.erase(data);
  }
}

void Cache::safeEraseFreeBlock(Block *block) {
  if (!block->allocated) {
    if (block->cache_type == CacheType::CACHE) {
      auto it = free_cache_blocks.find(block);
      if (it != free_cache_blocks.end()) {
        free_cache_blocks.erase(it);
      }
    } else {
      auto it = free_ldram_blocks.find(block);
      if (it != free_ldram_blocks.end()) {
        free_ldram_blocks.erase(it);
      }
    }
  }
}

void Cache::safeInsertFreeBlock(Block *block) {
  if (block->cache_type == CacheType::CACHE) {
    free_cache_blocks.insert(block);
  } else {
    free_ldram_blocks.insert(block);
  }
}

void Cache::peekFreeBlocks(CacheType type) {
  if (type == CacheType::CACHE) {
    for (auto block : free_cache_blocks) {
      LOG(INFO) << "Free cache block: " + TO_STRING(*block);
    }
  } else {
    for (auto block : free_ldram_blocks) {
      LOG(INFO) << "Free ldram block: " + TO_STRING(*block);
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
      safeEraseFreeBlock(replacee);
      // naive split to two blocks
      replacee->block_size = data_size;
      replacee->data = replacer_data;
      Block *remainder =
          new Block(false, replacee->block_offset + data_size, remainder_size,
                    replacee->next, replacee, replacee->cache_name,
                    replacee->cache_type, nullptr, -1);
      replacee->next->prev = remainder;
      replacee->next = remainder;

      // update free block list
      safeInsertFreeBlock(remainder);
      replacee->allocated = true;
    } else {
      // there is a following empty block, so we need to do merging
      Block *next_empty_block = replacee->next;
      replacee->allocated = true;
      replacee->block_size = data_size;
      replacee->data = replacer_data;

      safeEraseFreeBlock(next_empty_block);
      next_empty_block->block_offset = replacee->block_offset + data_size;
      next_empty_block->block_size += remainder_size;
      safeInsertFreeBlock(next_empty_block);
      // in this case, replacee is supposed to be allocated
      // no need to update free block list
    }
  } else if (remainder_size == 0) {
    // replace data in that block
    safeEraseFreeBlock(replacee);
    replacee->allocated = true;
    replacee->data = replacer_data;
  } else if (remainder_size < 0) {
    // To take place of a series of blocks
    // Only happens in cache
    safeEraseFreeBlock(replacee);
    Block *ptr = replacee->next;
    std::vector<Block *> blocks_to_delete;
    int64_t replacee_total_size = replacee->block_size;
    while (ptr->next != nullptr && replacee_total_size < data_size) {
      replacee_total_size += ptr->block_size;
      if (ptr->allocated) {
        replacee_data_list.push_back(ptr->data);
      } else {
        safeEraseFreeBlock(ptr);
      }
      blocks_to_delete.push_back(ptr);
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
    // delete memory of inbetween blocks
    for (auto block : blocks_to_delete) {
      delete block;
    }
  }
  return replacee_data_list;
}

void Cache::freeBlock(Block *target) {
  if (!target->allocated) {
    return;
  }
  target->allocated = false;
  target->data = nullptr;
  std::vector<Block *> blocks_to_delete;
  if (!target->next->allocated && target->prev->allocated) {
    target->block_size += target->next->block_size;
    blocks_to_delete.push_back(target->next);
    safeEraseFreeBlock(target->next);
    target->next = target->next->next;
    target->next->prev = target;
  } else if (target->next->allocated && !target->prev->allocated) {
    target->block_size += target->prev->block_size;
    target->block_offset = target->prev->block_offset;
    blocks_to_delete.push_back(target->prev);
    safeEraseFreeBlock(target->prev);
    target->prev = target->prev->prev;
    target->prev->next = target;
  } else if (!target->next->allocated && !target->prev->allocated) {
    target->block_size += target->next->block_size;
    target->block_size += target->prev->block_size;
    target->block_offset = target->prev->block_offset;
    blocks_to_delete.push_back(target->next);
    blocks_to_delete.push_back(target->prev);
    safeEraseFreeBlock(target->next);
    safeEraseFreeBlock(target->prev);
    target->next = target->next->next;
    target->next->prev = target;
    target->prev = target->prev->prev;
    target->prev->next = target;
  }
  for (auto block : blocks_to_delete) {
    delete block;
  }
  safeInsertFreeBlock(target);
}

int64_t Cache::free(CacheData *target_data) {
  LOG(INFO) << "Freeing cache memory for " + TO_STRING(*target_data) + "...";

  if (lockedData.count(*(target_data)) > 0) {
    // should not free any block with locked data
    return -1;
  }

  int64_t offset = -1;
  Block *ptr = cache_head->next;
  while (ptr->next != nullptr) {
    if (!ptr->allocated) {
      ptr = ptr->next;
      continue;
    }
    if (*(ptr->data) == (*target_data)) {
      offset = ptr->block_offset;
      freeBlock(ptr);
      break;
    }
  }
  return offset;
}

CacheHit Cache::load(CacheData *target_data) {
  int64_t size = target_data->size;
  if (size > cache_size) {
    LOG(ERROR) << "Cache size is less than data size.";
    return CacheHit(CacheHitLocation::ERROR);
  }

  bool match_cache = false;
  bool match_ldram = false;
  std::string result = "";
  Block *ptr = cache_head->next;
  std::string target_data_info = TO_STRING(*target_data);

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
    if (*(ptr->data) == (*target_data)) {
      // OUTCOME 0: found in cache
      match_cache = true;
      std::string cache_info = TO_STRING(*ptr);
      LOG(INFO) << indentation(1) + "Cache hit, " + target_data_info +
                       " has been cached on " + cache_info;
      updateBlockCount(ptr, true);
      target_cache_block = ptr;
      // need to update counts for all allocated blocks
      // should not break early
      // break;
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
    // allocate memory in cache
    // either find an empty cache block or find one to replace
    target_cache_block = cacheAlloc(target_data, 2);
    if (target_cache_block == nullptr) {
      return CacheHit(CacheHitLocation::ERROR);
    }

    // Then check ldram
    if (storedInLdram.count(*target_data) > 0) {
      Block *ldram_ptr = ldram_head->next;
      while (ldram_ptr->next != nullptr) {
        if (!ldram_ptr->allocated) {
          ldram_ptr = ldram_ptr->next;
          continue;
        }
        if (*(ldram_ptr->data) == (*target_data)) {
          // OUTCOME 1: found in ldram
          match_ldram = true;
          ldram_from_block = ldram_ptr;
          std::string cache_info = TO_STRING(*ldram_from_block);
          LOG(INFO) << indentation(3) + "LDRAM hit, " + target_data_info +
                           " has been previously stored in " +
                           TO_STRING(*ldram_from_block);
          break;
        }
        ldram_ptr = ldram_ptr->next;
      }
    }

    // No matter if data is found in ldram, a ldram_to_block is supposed to
    // be allocated for cache swapping. We also assume that ldram is large
    // enough so abort in case of any overflow
    Block *cmp_ldram = new Block(
        true, 0,
        std::max(target_cache_block->block_size, data_size) -
            (cache_align_size - 1),
        nullptr, nullptr, "", CacheType::LDRAM, target_cache_block->data, -1);
    auto ldram_block_it = free_ldram_blocks.lower_bound(cmp_ldram);
    if (ldram_block_it != free_ldram_blocks.end()) {
      ldram_to_block = *ldram_block_it;
      LOG(INFO) << indentation(3) + "Find LDRAM block " +
                       TO_STRING(*ldram_to_block) +
                       " to store the replaced data " +
                       TO_STRING(*target_cache_block) + " from cache.";
    } else {
      LOG(ERROR) << "LDRAM has no more space.";
      return CacheHit(CacheHitLocation::ERROR);
    }
    delete cmp_ldram;
  }
  // Now that we already know cache/ldram_from/ldram_to locations, we then
  // update the information in the block link list
  if (match_cache) {
    // OUTCOME 0: found in cache
    return CacheHit(CacheHitLocation::CACHE, target_cache_block->block_offset);
  } else {
    // OUTCOME 1/2: not found in cache
    // a. load target to cache
    auto replacee_data_list = loadData2Block(target_data, target_cache_block);
    initBlockCount(target_cache_block);
    // b. load replaced cache data to ldram
    std::vector<int64_t> ldram_to_block_list;
    std::vector<int64_t> replaced_data_size_list;
    if (!replacee_data_list.empty()) {
      for (auto replacee_data : replacee_data_list) {
        // write back to ldram in an consistent manner
        loadData2Block(replacee_data, ldram_to_block);
        storedInLdram.insert(*replacee_data);
        ldram_to_block_list.push_back(ldram_to_block->block_offset);
        replaced_data_size_list.push_back(replacee_data->size);
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
                        ldram_from_block->block_offset, ldram_to_block_list,
                        replaced_data_size_list);
      } else {
        // OUTCOME 1.1: found in ldram, cache has space
        return CacheHit(CacheHitLocation::LDRAM,
                        target_cache_block->block_offset,
                        ldram_from_block->block_offset);
      }
    } else {
      // OUTCOME 2: not found at all
      if (!replacee_data_list.empty()) {
        // OUTCOME 2.1: not found at all, cache full
        return CacheHit(CacheHitLocation::NOT_FOUND,
                        target_cache_block->block_offset, -1,
                        ldram_to_block_list, replaced_data_size_list);
      } else {
        // OUTCOME 2.2: not found at all, cache has space
        return CacheHit(CacheHitLocation::NOT_FOUND,
                        target_cache_block->block_offset);
      }
    }
  }
}

Block *Cache::cacheAlloc(CacheData *target_data, int indent) {
  Block *target_cache_block = nullptr;
  int64_t data_size = PAD_UP(target_data->size, cache_align_size);
  // Find an empty cache block first
  Block *cmp = new Block(true, 0, data_size - (cache_align_size - 1), nullptr,
                         nullptr, "", CacheType::CACHE, target_data, -1);
  auto cache_block_it = free_cache_blocks.lower_bound(cmp);
  if (cache_block_it != free_cache_blocks.end()) {
    // found an empty cache block
    target_cache_block = *cache_block_it;
    LOG(INFO) << indentation(indent) + "Find an empty block " +
                     TO_STRING(*target_cache_block) + " to cache " +
                     TO_STRING(*target_data) + ".";
  } else {
    // Cache full, find one cache block to swap
    Block *ptr = cache_head->next;
    while (ptr->next != nullptr) {
      if (!ptr->allocated) {
        ptr = ptr->next;
        continue;
      }
      if (target_cache_block == nullptr) {
        target_cache_block = ptr;
        break;
      }
      ptr = ptr->next;
    }
    while (ptr->next != nullptr) {
      if (ptr->allocated && cacheReplaceable(ptr, target_cache_block) &&
          cache_size - ptr->block_offset >= target_data->size) {
        target_cache_block = ptr;
      }
      ptr = ptr->next;
    }
    if (target_cache_block == nullptr) {
      LOG(ERROR) << "Cache has no more space for " + TO_STRING(*target_data) +
                        ".";
      return nullptr;
    }
    // try best to minimize fragments
    if (!target_cache_block->prev->allocated) {
      target_cache_block = target_cache_block->prev;
    }
    LOG(INFO) << indentation(indent) + "Cache Full. Find cache block " +
                     TO_STRING(*target_cache_block) + " to replace.";
  }
  delete cmp;
  return target_cache_block;
}

CacheHit Cache::allocate(CacheData *target_data) {
  int64_t size = target_data->size;
  if (size > cache_size) {
    LOG(ERROR) << "Cache size is less than data size.";
    return CacheHit(CacheHitLocation::ERROR);
  }

  LOG(INFO) << "Allocating cache memory for " + TO_STRING(*target_data) + "...";

  // Cache block to return
  Block *target_cache_block = cacheAlloc(target_data, 1);

  if (target_cache_block == nullptr) {
    return CacheHit(CacheHitLocation::ERROR);
  } else {
    return CacheHit(CacheHitLocation::NOT_FOUND,
                    target_cache_block->block_offset);
  }
}

void Cache::printMemoryGraph(Block *head, int height = 16, int width = 64) {
  int64_t total =
      (head->cache_type == CacheType::CACHE) ? cache_size : ldram_size;
  std::string info_string =
      " " + TO_STRING(head->cache_type) + " of " + head->cache_name;
  info_string += ", Size: " + std::to_string(total) + " ";
  info_string = lrpad(info_string, width + 12, '=');
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
        output = lpad(std::to_string(cur_row * row_size), 5, ' ') + " | ";
        cur_col = 0;
      }
    }
    ptr = ptr->next;
  }
}

void Cache::printInformation() {
  std::string info_string = "";
  info_string += ">>> Cache ";
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
  LOG(INFO) << "";
  printMemoryGraph(cache_head);
  printBlocks(cache_head);
  LOG(INFO) << "";
  printMemoryGraph(ldram_head);
  printBlocks(ldram_head);
  LOG(INFO) << "";
}

void Cache::printBlocks(Block *head) {
  std::string info_string =
      std::string("  -- Data in " + TO_STRING(head->cache_type) + ": ");
  LOG(INFO) << info_string;
  Block *ptr = head->next;
  while (ptr->next != nullptr) {
    info_string = indentation(4) + "- Block ";
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

}  // namespace infini
