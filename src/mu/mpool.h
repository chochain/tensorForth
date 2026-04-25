/**
 * @file
 * @brief TLSF class - tensor storage manager interface
 *
 *  <pre>Copyright (C) 2019 GreenII. This file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __MU_MPOOL_H
#define __MU_MPOOL_H
#pragma once
#include <stddef.h>
#include <mutex>
#include <string>
#include "ten4_config.h"

#if T4_DO_OBJ                                  /// * only when object is activated

namespace t4::mu {
// ============================================================================
// MemoryPool
//
// Singleton fixed-size (max(Tensor/Model/Dataset) = 160-byte) memory pool.
// Storage is heap-allocated via malloc on first get_instance() call.
//
// Layout:
//  malloc'd buffer
//  ┌───────────┬───────────┬─────┬───────────┐
//  │  Block 0  │  Block 1  │ ... │ Block N-1 │
//  └───────────┴───────────┴─────┴───────────┘
//  Each free block's first sizeof(void*) bytes hold a pointer to the
//  next free block (intrusive singly-linked free list).
// ============================================================================
class Mpool {
    int    _bsz;                             ///< block size
    int    _nblock;                          ///< max number of blocks
    char*  _storage;                         ///< malloc'd flat buffer
    void*  _free_head;                       ///< head of the free list
    int    _alloc_cnt;

    mutable std::mutex _mutex;               ///< multi-threading control
    
    Mpool() : _storage(nullptr), _free_head(nullptr), _alloc_cnt(0) {}
    ~Mpool() {                               ///< frees the malloc'd buffer
        std::free(_storage);                 /// * release back to OS 
        _storage = nullptr;
    }
    
public:
    /// ------------------------------------------------------------------
    /// Singleton access — creates the pool on first call.
    /// ------------------------------------------------------------------
    static Mpool &get_instance();
    /// ------------------------------------------------------------------
    /// Core API
    /// ------------------------------------------------------------------
    void  init(int bsz, int nblock);          ///< construction only via get_instance()
    void  *malloc();                          ///< Returns a block. Throws std::bad_alloc if exhausted.
    void  free(void *ptr);                    ///< Returns a block to the pool. No-op on nullptr.
    int   offset(void *ptr) const { return (char*)ptr - _storage; } ///< ptr offset to _storage
    void  status();                           ///< for sanity check
    ///
    /// Non-copyable, non-movable.
    ///
    Mpool(const Mpool&)            = delete;
    Mpool& operator=(const Mpool&) = delete;
    Mpool(Mpool&&)                 = delete;
    Mpool& operator=(Mpool&&)      = delete;
    ///
    /// sanity check
    ///
    ///
    /// object API
    ///
    template <typename T, typename... Args>   /// Typed helpers: alloc + construct / destruct + free
    T* create(Args&&... args) {
        static_assert(sizeof(T) <= _bsz,
                      "T is larger than BLK_SZ (160 bytes)");
        void* ptr = malloc();
        try         { return ::new (ptr) T(std::forward<Args>(args)...); }
        catch (...) { free(ptr); throw; }
    }

    template <typename T>
    void destroy(T* ptr) noexcept {
        if (!ptr) return;
        ptr->~T();
        free(ptr);
    }
    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
    int      alloc_cnt() const { std::unique_lock<std::mutex> lock(_mutex); return _alloc_cnt; }
    int      free_cnt()  const { return _nblock - alloc_cnt(); }
    bool     is_full()   const { return free_cnt() == 0;       }
    bool     is_empty()  const { return alloc_cnt() == 0;      }
    bool     is_own(const void* ptr) const {
        const uintptr_t p  = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t s0 = reinterpret_cast<uintptr_t>(_storage);
        const uintptr_t s1 = s0 + _bsz * _nblock;
        return p >= s0 && p < s1 && (p - s0) % _bsz == 0;
    }

private:
    void* _block(int i) const {
        return static_cast<void*>(_storage + i * _bsz);
    }
};

} // namespace t4::mu

#endif // T4_DO_OBJ
#endif // __MU_MPOOL_H
