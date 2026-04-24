/** -*- c++ -*-
 * @file
 * @brief MPOOL class - host fixed-size (max(Tensor/Model/Dataset) = 160B) memory pool
 *
 * <pre>Copyright (C) 2022 GreenII. This file is distributed under BSD 3-Clause License.</p>
 */
#include <cstdlib>         // malloc, free
#include <cstdint>         // uintptr_t
#include <cassert>
#include <stdexcept>
#include <new>             // std::bad_alloc
#include <string>
#include "mpool.h"

namespace t4::mu {

// ============================================================================
// private Constructor / Destructor
// ============================================================================
Mpool::Mpool()
    : _storage(nullptr), _free_head(nullptr), _alloc_cnt(0)
{
    _storage = static_cast<char*>(malloc(BLK_SZ * BLK_CNT));
    if (!_storage) throw std::bad_alloc{};

    for (int i = 0; i < BLK_CNT - 1; ++i) { /// setup free list, each point to next
        void* here = _block(i);
        void* next = _block(i + 1);
        *reinterpret_cast<void**>(here) = next;
    }
    *reinterpret_cast<void**>(_block(BLK_CNT - 1)) = nullptr;  ///< end of list
    
    _free_head = _block(0);
}

// ============================================================================
// Core API
// ============================================================================
void* Mpool::alloc() {
    std::unique_lock<std::mutex> lock(_mutex);

    if (!_free_head) throw std::bad_alloc{};

    void* blk = _free_head;
    _free_head  = *reinterpret_cast<void**>(_free_head);
    ++_alloc_cnt;
    return blk;
}

void Mpool::free(void* ptr) {
    if (!ptr) return;

    assert(is_own(ptr) && "Mpool::free: pointer does not belong to this pool");

    std::unique_lock<std::mutex> lock(_mutex);

    *reinterpret_cast<void**>(ptr) = _free_head;
    _free_head = ptr;
    _alloc_cnt--;
}

std::string Mpool::status() const {
    std::unique_lock<std::mutex> lock(_mutex);
    return std::string("Mpool(160B x 1024)  allocated=")
         + std::to_string(_alloc_cnt)
         + "  free=" + std::to_string(BLK_CNT - _alloc_cnt);
}

} // namespace t4::mu
