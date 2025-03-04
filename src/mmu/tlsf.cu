/** -*- c++ -*-
 * @file
 * @brief TLSF class - tensor storage manager implementation
 *
 * <pre>Copyright (C) 2022 GreenII. This file is distributed under BSD 3-Clause License.</p>
*/
#include "ten4_types.h"
#include "util.h"
#include "tlsf.h"

#if T4_DO_OBJ
// TLSF: Two-Level Segregated Fit allocator with O(1) time complexity.
// Layer 1st(f), 2nd(s) model, smallest block 16-bytes, 16-byte alignment
// TODO: multiple-pool, thread-safe
// semaphore
#define _LOCK           { MUTEX_LOCK(_mutex); }
#define _UNLOCK         { MUTEX_FREE(_mutex); }
#define U8PADD(p, n)	((U8*)(p) + (n))                    /** pointer add */
#define U8PSUB(p, n)	((U8*)(p) - (n))                    /** pointer sub */
#define U8POFF(p1, p0)	((S32)((U8*)(p1) - (U8*)(p0)))      /** calc offset */
#define TADDR(p)        ((U32)((U8*)(p) - _heap))           /** heap offset */

//================================================================
/*! constructor

  @param  ptr    pointer to free memory block.
  @param  size    size. (max 4G)
*/
__BOTH__ void
TLSF::init(U8 *mem, U64 sz, U64 off) {
    TRACE("\\ TLSF: ostore=%p, alloc=0x%lx", mem, sz);
    _heap    = mem + off;                               // header offset (for Tensor0)
    _heap_sz = sz - off;
    U64 bsz  = _heap_sz - sizeof(used_block);           // minus end block
    //
    // clean TLSF maps
    //
    for (int i=0; i<L1_BITS; i++)  _l2_map[i]    = 0;
    for (int i=0; i<FL_SLOTS; i++) _free_list[i] = 0;
    //
    // initialize entire memory pool as the first block
    //
    free_block *head  = (free_block*)_heap;
    head->bsz  = bsz;                                   // 1st (big) block
    head->psz  = 0;
    head->next = head->prev = 0;
    SET_FREE(head);
    //
    // get max available index
    //
    long i = 31L; for (U64 z = bsz, m = 1L<<31; i && z && !(z & m); z<<=1) i--;
    long j = (bsz >> (i - L2_BITS)) & L2_MASK;
    U32 index = INDEX(i, j);                            // last slot of map
    TRACE(" => bsz=0x%lx, index(%lx,%lx)\n", bsz, i, j);
    SET_MAP(index);                                     // set ticks for available maps
    _free_list[index] = head;

    used_block *tail = (used_block*)BLK_AFTER(head);    // last block
    tail->bsz = 0;
    tail->psz = bsz;
    SET_USED(tail);
}

//================================================================
/*! allocate memory

  @param  size    request storage size.
  @return void* pointer to a guru memory block.
*/
__GPU__ void*
TLSF::malloc(U64 sz) {
    MM_DB("  tlsf#malloc(0x%lx) {\n", sz);
    U64 bsz = ALIGN8(sz) + sizeof(used_block);  // logical => physical size

    _LOCK;
    U32 index       = _find_free_index(bsz);
    free_block *blk = _set_used(index);         // take the indexed block off free list

    _split(blk, bsz);                           // allocate the block, free up the rest
    _UNLOCK;

    ASSERT(blk->bsz >= bsz);                    // make sure it provides big enough a block

    void *data = BLK_DATA(blk);
    MM_DB("  } tlsf#malloc => %x:%lx\n", TADDR(data), sz);
    return data;                                // pointer to raw space
}

//================================================================
/*! re-allocate memory

  @param  ptr    Return value of raw malloc()
  @param  size    request size
  @return void* pointer to allocated memory.
*/
__GPU__ void*
TLSF::realloc(void *p0, U64 sz) {
    ASSERT(p0);
    U64 bsz = ALIGN8(sz) + sizeof(used_block);           // include the header

    used_block *blk = (used_block *)BLK_HEAD(p0);
    ASSERT(IS_USED(blk));                                // make sure it is used

    if (bsz > blk->bsz) {
        _merge_next((free_block *)blk);                  // try to get the block bigger
    }
    if (bsz == blk->bsz) return p0;                      // fits right in
    if ((blk->bsz > bsz) &&
            ((blk->bsz - bsz) > T4_STRBUF_SZ))   {       // split a really big block
        _LOCK;
        _split((free_block*)blk, bsz);
        _UNLOCK;
        return p0;
    }
    //
    // compacting, mostly for str buffer
    // instead of splitting, since Ruby reuse certain sizes
    // it is better to allocate a block and release the original one
    //
    void *ret = this->malloc(bsz);
    MEMCPY(ret, (const void*)p0, (size_t)sz);            // deep copy, !!using CUDA provided memcpy
    this->free(p0);                                      // reclaim block

    return ret;
}

//================================================================
/*! release memory
*/
__GPU__ void
TLSF::free(void *ptr) {
    if (!ptr) return;

    _LOCK;
    const U32 addr  = TADDR(ptr);
    free_block *blk = (free_block *)BLK_HEAD(ptr);       // get block header
    MM_DB("  tlsf#free(%x) %x:%x {\n", addr, TADDR(blk), blk->bsz);
    _merge_next(blk);
    _set_free(blk);

    // the block is free now, try to merge a free block before if exists
    _merge_prev(blk);
    MM_DB("  } tlsf#free(%x)\n", addr);
    _UNLOCK;
}

//================================================================
/*! calc l1 and l2, and returns fli,sli of free_blocks

  @param  alloc_size    alloc size
  @retval int            index of free_blocks

  original thesis:
    l1 = fls(sz);
    l2 = (sz ^ (1<<l1)) >> (l1 - L2_BITS);  // 2 shifts, 1 minus, 1 xor

  mrbc: ???
    l1 = __fls(sz >> (MN_BITS + L2_BITS);
    n  = (l1==0) ? (l1 + MN_BITS) : (l1 + MN_BITS - 1);
    l2 = (sz >> n) & L2_MASKS;
*/
//================================================================
// find last set bit, i.e. most significant bit (0-31)
__GPU__ U32
TLSF::_idx(U64 sz) {
    auto __fls = [](U32 x) {
        U32 n;
        asm("bfind.u32 %0, %1;\n\t" : "=r"(n) : "r"(x));
        return n;
    };
    U32 l1 = __fls(sz);
    U32 l2 = (sz >> (l1 - L2_BITS)) & L2_MASK;    // 1 shift, 1 minus, 1 and

    MM_DB("    tlsf#idx(%lx) INDEX(%x,%x) => %x\n", sz, l1, l2, INDEX(l1, l2));

    return INDEX(l1, l2);
}

//================================================================
/*! Find index to a free block

  @param  size    number of minimal unit
  @retval -1    not found
  @retval index to available _free_list
*/
__GPU__ S32
TLSF::_find_free_index(U64 sz) {
    U32 index = _idx(sz);                        // find free_list index by size

    if (_free_list[index]) return index;         // free block readily available

    // no previous block exist, create a new one
    U32 l1 = L1(index);
    U32 l2 = L2(index);
    U32 m1, m2 = _l2_map[l1] >> (l2+1);          // get SLI one size bigger
    MM_DB("    tlsf#find(%x) l2_map[%x]=%x", index, l1, _l2_map[l1]);
    if (m2) {                                    // check if any 2nd level slot available
        l2 = __ffs(m2 << l2);                    // MSB represent the smallest slot that fits
    }
    else if ((m1 = (_l1_map >> (l1+1))) != 0) {  // get FLI one size bigger
        l1 = __ffs(m1 << l1);                    // allocate lowest available bit
        l2 = __ffs(_l2_map[l1]) - 1;             // get smallest size
    }
    else {
        l1 = l2 = 0xff;                          // out of memory
    }
    MM_DB(", (m1|m2)=%x|%x, INDEX(%x,%x) => %x\n", m1, m2, l1, l2, INDEX(l1, l2));

    return INDEX(l1, l2);                        // index to freelist head
}

//================================================================
/*! Split free block by size (before allocating)

  @param  blk    pointer to free block
  @param  size    storage size
*/
__GPU__ void
TLSF::_split(free_block *blk, U64 bsz) {
    ASSERT(IS_USED(blk));

    U64 minsz = ALIGN8(bsz) + (1 << MN_BITS) + 2*sizeof(free_block);
    if (blk->bsz < minsz) return;                                     // too small to split

    // split block, free
    free_block *free = (free_block *)U8PADD(blk, bsz);                // future next block (i.e. alot bsz bytes)

    MM_DB("    tlsf#split(%x:%x,%lx) => ", TADDR(blk), blk->bsz, bsz);
    free->bsz = blk->bsz - bsz;                                       // carve out the acquired block
    free->psz = U8POFF(free, blk);                                    // positive offset to previous block
    blk->bsz  = bsz;                                                  // allocate target block
    MM_DB("%x:%x + %x:%x\n", TADDR(blk), blk->bsz, TADDR(free), free->bsz);

    free_block *aft  = (free_block *)BLK_AFTER(blk);                  // next adjacent block
    if (aft) {
        aft->psz = U8POFF(aft, free) | (aft->psz & FREE_FLAG);        // backward offset (positive)
        _merge_next(free);                                            // _combine if possible
    }
    _set_free(free);            // add to free_list and set (free, tail, next, prev) fields
}

//================================================================
/*! merge p0 and p1 adjacent free blocks.
  ptr2 will disappear

  @param  ptr1    pointer to free block 1
  @param  ptr2    pointer to free block 2
*/
__GPU__ void
TLSF::_pack(free_block *b0, free_block *b1) {
    ASSERT((free_block*)BLK_AFTER(b0)==b1);
    ASSERT(IS_FREE(b1));

    // remove b0, b1 from free list first (sizes will not change)
    _unmap(b1);

    MM_DB("    tlsf#pack(%x:%x + %x:%x) => ", TADDR(b0), b0->bsz, TADDR(b1), b1->bsz);
    // merge b0 and b1, retain b0.FREE_FLAG
    used_block *b2 = (used_block *)BLK_AFTER(b1);
    b2->psz += b1->psz & ~FREE_FLAG;    // watch for the block->flag
    b0->bsz += b1->bsz;                 // include the block header

    MM_DB("%x:%x\n", TADDR(b0), b0->bsz);
}

//================================================================
/*! wipe the free_block from linked list

  @param  blk    pointer to free block.
*/
__GPU__ void
TLSF::_unmap(free_block *blk, U32 bidx) {
    MM_DB("    tlsf#unmap(%x:%x,%x)\n", TADDR(blk), blk->bsz, bidx);
    ASSERT(IS_FREE(blk));                        // ensure block is free

    U32 index = bidx ? bidx : _idx(blk->bsz);
    free_block *n = _free_list[index] = NEXT_FREE(blk);
    free_block *p = blk->prev ? PREV_FREE(blk) : NULL;
    if (n) {                                     // up link
        // blk->next->prev = blk->prev;
        if (blk->prev) {
            n->prev = U8POFF(p, n);
            ASSERT((n->prev&7)==0);
            SET_FREE(n);
        }
        else n->prev = 0;
    }
    else {                                       // 1st of the link
        CLEAR_MAP(index);                        // clear the index bit
    }
    if (blk->prev) {                             // down link
        // blk->prev->next = blk->next;
        p->next = blk->next ? U8POFF(n, p) : 0;
    }
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  blk    Pointer to block to be freed.

  TODO: check thread safety
*/
__GPU__ void
TLSF::_set_free(free_block *blk) {
    ASSERT(IS_USED(blk));

    U32 index = _idx(blk->bsz);
    U32 l1 = L1(index);
    U32 l2 = L2(index);
    _l1_map     |= TIC(l1);
    _l2_map[l1] |= TIC(l2);
//    SET_MAP(index);                                // set ticks for available maps

    // update block attributes
    free_block *head = _free_list[index];
    MM_DB("    tlsf#set_free(<%x> => %x:%x)\n", index, TADDR(blk), blk->bsz);

    SET_FREE(blk);
    blk->next = head ? U8POFF(head, blk) : 0;     // setup linked list
    blk->prev = 0;
    if (head) {                                   // non-end block, add backward link
        head->prev = U8POFF(blk, head);
        SET_FREE(head);                           // turn the free flag back on
    }
    _free_list[index] = blk;                      // new head of the linked list
}

__GPU__ free_block*
TLSF::_set_used(U32 index) {
    MM_DB("    tlsf#set_used(<%x>)\n", index);
    free_block *blk  = _free_list[index];
    ASSERT(blk);
    ASSERT(IS_FREE(blk));

    _unmap(blk, index);
    SET_USED(blk);

    return blk;
}

__GPU__ void
TLSF::_merge_next(free_block *b0) {
    free_block *b1 = (free_block *)BLK_AFTER(b0);
    MM_DB("    tlsf#merge_next %x:%x + %x:%x.%s\n",
          TADDR(b0), b0->bsz, TADDR(b1), b1->bsz,
          IS_FREE(b1) ? "free" : "used");
    while (b1 && IS_FREE(b1) && b1->bsz!=0) {
        _pack(b0, b1);
        b1 = (free_block *)BLK_AFTER(b0);    // try the already expanded block again
    }
}

__GPU__ free_block*
TLSF::_merge_prev(free_block *b1) {
    free_block *b0 = (free_block *)BLK_BEFORE(b1);
    MM_DB("    tlsf#merge_prev %x:%x.%s + %x:%x:%x\n",
          b0 ? TADDR(b0) : 0,
          b0 ? b0->bsz   : 0,
          b0 ? (IS_FREE(b0) ? "free" : "used") : "head",
          TADDR(b1), b1->bsz, b1->psz);

    if (b0==NULL || IS_USED(b0)) return b1;
    _unmap(b0);                              // take it out of free_list before merge
    _pack(b0, b1);                           // take b1 out and merge with b0

    SET_USED(b0);                            // _set_free assume b0 to be a USED block
    _set_free(b0);

    return b0;
}
//================================================================
// MMU JTAG sanity check - memory pool walker
//
//================================================================
__BOTH__ int
TLSF::_mmu_ok()    {                         // mmu sanity check
    used_block *p0 = (used_block*)_heap;
    used_block *p1 = (used_block*)BLK_AFTER(p0);
    U32 tot = sizeof(free_block);
    while (p1) {
        if (p0->bsz != (p1->psz&~FREE_FLAG)) {       // ERROR!
            return 0;                                // memory integrity broken!
        }
        tot += p0->bsz;
        p0  = p1;
        p1  = (used_block*)BLK_AFTER(p0);
    }
    return (tot==_heap_sz) && (!p1);         // last check
}
__BOTH__ void
TLSF::_show_stat() {
#if MM_DEBUG
    ///
    /// stat pre-adjusted for the stopper block
    ///
    int tot=sizeof(used_block), free=0, used=0;
    int nblk=-1, nused=-1, nfree=0, nfrag=0;

    used_block *p = (used_block*)_heap;
    U32 f0 = IS_FREE(p);                  // starting block type
    while (p) {                           // walk the memory pool
        U64 bsz = p->bsz;                 // current block size
        tot   += bsz;
        nblk  += 1;
        if (IS_FREE(p)) {
            nfree += 1;
            free  += bsz;
            if (!f0) nfrag++;             // is adjacent block fragmented
        }
        else {
            nused += 1;
            used  += bsz;
        }
        f0 = IS_FREE(p);
        p  = (used_block*)BLK_AFTER(p);
    }
    float pct = 100.0*used/tot;

    INFO("\\ OBJ: used[%d]=%d(0x%x) %.2f%% allocated", nused, used, used, pct);
    INFO(" free[%d]=%d(0x%x), total=%d(0x%x)", nfree, free, free, tot, tot);
    INFO(" nblk=%d, nfrag=%d\n", nblk, nfrag);
#endif // MM_DEBUG
}

__BOTH__ void
TLSF::_dump_freelist() {
#if MM_DEBUG
    INFO("  tlsf#L1=%4x: ", _l1_map);
    for (int i=L1_BITS-1;  i>=0; i--) {
        INFO("%02x%s", _l2_map[i], i%4==0 ? " " : "");
    }
    for (int i=FL_SLOTS-1; i>=0; i--) {
        if (!_free_list[i]) continue;
        INFO("\n\t<%02x>=>[", i);
        for (free_block *b = _free_list[i]; b!=NULL; b=NEXT_FREE(b)) {
            INFO(" %x:%x", TADDR(b), b->bsz);
            if (IS_USED(b)) {
                INFO("<-USED?");
                break;                // something is wrong (link is broken here)
            }
        }
        INFO(" ] ");
    }
    INFO("\n");
#endif // MM_DEBUG
}

#endif // T4_DO_OBJ
