/**
 * @file
 * @brief TLSF class - tensor storage manager interface
 *
 *  <pre>Copyright (C) 2019 GreenII. This file is distributed under BSD 3-Clause License.</pre>
 */
#if (!defined(__MMU_MPOOL_H) && T4_DO_OBJ)
#define __MMU_MPOOL_H
#pragma once
#include <stddef.h>
#include <mutex>
#include <string>

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
public:
    static const int BLK_SZ  = 160;
    static const int BLK_CNT = 1024;

    /// ------------------------------------------------------------------
    /// Singleton access — creates the pool on first call.
    /// ------------------------------------------------------------------
    static void *get_instance() {
        static Mpool pool0;                   ///< constructed once, destroyed at program exit
        return (void*)&pool0;
    }
/*
    /// Non-copyable, non-movable.
    Mpool(const Mpool&)            = delete;
    Mpool& operator=(const Mpool&) = delete;
    Mpool(Mpool&&)                 = delete;
    Mpool& operator=(Mpool&&)      = delete;
*/
    /// ------------------------------------------------------------------
    /// Core API
    /// ------------------------------------------------------------------
    void* alloc();                            ///< Returns a block. Throws std::bad_alloc if exhausted.
    void  free(void *ptr);                    ///< Returns a block to the pool. No-op on nullptr.
    int   offset(void *ptr) { return (char*)ptr - _storage; } ///< ptr offset to _storage

    template <typename T, typename... Args>   /// Typed helpers: alloc + construct / destruct + free
    T* create(Args&&... args) {
        static_assert(sizeof(T) <= BLK_SZ,
                      "T is larger than BLK_SZ (160 bytes)");
        void* ptr = alloc();
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
    int      free_cnt()  const { return BLK_CNT - alloc_cnt(); }
    bool     is_full()   const { return free_cnt() == 0;       }
    bool     is_empty()  const { return alloc_cnt() == 0;      }
    bool     is_own(const void* ptr) const {
        const uintptr_t p  = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t s0 = reinterpret_cast<uintptr_t>(_storage);
        const uintptr_t s1 = s0 + BLK_SZ * BLK_CNT;
        return p >= s0 && p < s1 && (p - s0) % BLK_SZ == 0;
    }
    std::string status();

private:
    Mpool();                                  ///< construction only via get_instance()
    ~Mpool() {                                ///< frees the malloc'd buffer
        free(_storage);
        _storage = nullptr;
    }
    void* _block(int i) const {
        return static_cast<void*>(_storage + i * BLK_SZ);
    }
    char*  _storage;                         ///< malloc'd flat buffer
    void*  _free_head;                       ///< head of the free list
    int    _alloc_cnt;

    mutable std::mutex _mutex;               ///< multi-threading control
};

} // namespace t4::mu

#endif // (!defined(__MMU_MPOOL_H) && T4_DO_OBJ)
