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
#include "ten4_types.h"
#include "mpool.h"

#if T4_DO_OBJ              /// * only when object system is activated

namespace t4::mu {
///
/// multi-threading lock
///
#define LOCK()    std::unique_lock<std::mutex> lock(_mutex)
#define UNLOCK()  lock.unlock()
#define OFFSET(p) ((int)((char*)(p) - _storage))           /** storage offset */

Mpool &Mpool::get_instance() {
    static Mpool pool0;                   ///< constructed once, destroyed at program exit
    return pool0;
}

// ============================================================================
// private Constructor / Destructor
// ============================================================================
void *Mpool::init(int bsz, int nblock) {
    LOCK();
    _bsz       = bsz;
    _nblock    = nblock;
    _storage   = static_cast<char*>(std::malloc(bsz * nblock));
    if (!_storage) throw std::bad_alloc{};

    for (int i = 0; i < nblock - 1; ++i) { /// setup free list, each point to next
        void* here = _block(i);
        void* next = _block(i + 1);
        *reinterpret_cast<void**>(here) = next;
    }
    *reinterpret_cast<void**>(_block(nblock - 1)) = nullptr;  ///< end of list
    
    _free_head = _block(0);

    return (void*)_storage;
}

// ============================================================================
// Core API
// ============================================================================
void *Mpool::malloc() {
    LOCK();
    MM_DB("  mpool#malloc() {\n");
    if (!_free_head) throw std::bad_alloc{};

    void* blk = _free_head;
    _free_head  = *reinterpret_cast<void**>(_free_head);
    ++_alloc_cnt;
    
    MM_DB("    mpool#alloc_cnt = %d\n"
          "  } mpoolf#malloc => %x:%x\n", _alloc_cnt, OFFSET(blk), _bsz);
    return blk;
}

void Mpool::free(void *ptr) {
    MM_DB("  mpool#free(%x) %d(0x%x) {\n", OFFSET(ptr), _bsz, _bsz);
    if (!ptr) return;

    assert(is_own(ptr) && "Mpool::dealloc: pointer does not belong to this pool");

    LOCK();

    *reinterpret_cast<void**>(ptr) = _free_head;
    _free_head = ptr;
    _alloc_cnt--;
    MM_DB("    mpool#alloc_cnt = %d\n"
          "  } mpool#free(%x)\n", _alloc_cnt, OFFSET(ptr));
}

void Mpool::status() {
    LOCK();
    INFO("\\ OBJ : used[%d] (fixed %dB), free[%d/%d]\n",
         _alloc_cnt, _bsz, (_nblock - _alloc_cnt), _nblock);
}

} // namespace t4::mu

#endif  // T4_DO_OBJ
