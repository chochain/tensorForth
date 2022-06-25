/*! @file
  @brief - tensorForth tensor memory storage management.

  <pre>Copyright (C) 2019 GreenII. This file is distributed under BSD 3-Clause License.</pre>
*/
#ifndef TEN4_SRC_TLSF_H_
#define TEN4_SRC_TLSF_H_

typedef struct used_block {          //< 8-bytes
    U32 bsz;                         //< block size, header included (max 2G)
    U32 psz;                         //< prior adjacent memory block size
} used_block;

typedef struct free_block {          //< 16-bytes (i.e. mininum allocation per block)
    U32 bsz;                         //< block size, header included (max 2G)
    U32 psz;                         //< prior adjacent memory block size
    S32 next;                        //< offset to next free block
    S32 prev;                        //< offset to previous free block
} free_block;

#define FREE_FLAG       0x1
#define IS_FREE(b)      ((b)->psz & FREE_FLAG)
#define IS_USED(b)      (!IS_FREE(b))
#define SET_FREE(b)     ((b)->psz |=  FREE_FLAG)
#define SET_USED(b)     ((b)->psz &= ~FREE_FLAG)

#define NEXT_FREE(b)    ((free_block*)((b)->next ? ((b)->next<0 ? U8PSUB((b), -(b)->next) : U8PADD((b), (b)->next)) : NULL))
#define PREV_FREE(b)    ((free_block*)((b)->prev ? ((b)->prev<0 ? U8PSUB((b), -(b)->prev) : U8PADD((b), (b)->prev)) : NULL))

#define BLK_AFTER(b)    (((b)->bsz           ) ? U8PADD(b, (b)->bsz             ) : NULL)        /**> following adjacent memory block  */
#define BLK_BEFORE(b)   (((b)->psz&~FREE_FLAG) ? U8PSUB(b, ((b)->psz&~FREE_FLAG)) : NULL)        /**> prior adjacent memory block      */
#define BLK_DATA(b)     (U8PADD(b, sizeof(used_block)))                                          /**> pointer to raw data space        */
#define BLK_HEAD(p)     (U8PSUB(p, sizeof(used_block)))                                          /**> block header from raw pointer    */

#define MN_BITS         4                            /**> 16 bytes minimal allocation */
#define L2_BITS         3                            /**> 8 entries                   */
#define L1_BITS         31                           /**> 31 levels, max 4G range     */

#define TIC(n)          (1 << (n))
#define L2_MASK         ((1<<L2_BITS)-1)             /**> level 2 bit mask */
#define MIN_BLOCK       (1 << MN_BITS)

#define L1(i)           ((i) >> L2_BITS)             /**> extrace L1 from given index, for bugging */
#define L2(i)           ((i) & L2_MASK)              /**> extract L2 from given index, for debugging */
#define FL_SLOTS        (L1_BITS * (1 << L2_BITS))   /**> slots for free_list pointers               */

#define L1_MAP(i)       (_l1_map)
#define L2_MAP(i)       (_l2_map[L1(i)])
#define INDEX(l1, l2)   (((l1)<<L2_BITS) | (l2))

#define SET_L1(i)       (L1_MAP(i) |= TIC(L1(i)))
#define SET_L2(i)       (L2_MAP(i) |= TIC(L2(i)))
#define SET_MAP(i)      { SET_L1(i); SET_L2(i); }
#define CLR_L1(i)       (L1_MAP(i) &= ~TIC(L1(i)))
#define CLR_L2(i)       (L2_MAP(i) &= ~TIC(L2(i)))
#define CLEAR_MAP(i)    { CLR_L2(i); if ((L2_MAP(i))==0) CLR_L1(i); }

class TLSF : public Managed {
    U8         *_heap;                  // CUDA kernel tensor storage memory pool
    U32        _heap_sz;
    U32        _mutex  = 0;
    U32        _l1_map = 0;             // use upper (FLI - MSB) bits
    U8         _l2_map[L1_BITS];        // 8-bit, (16-bit requires too many FL_SLOTS)
    free_block *_free_list[FL_SLOTS];   // array of free lists

public:
    __BOTH__ void        init(U8 *mem, U64 sz);
    __GPU__  void*       malloc(U32 sz);
    __GPU__  void*       realloc(void *p0, U32 sz);
    __GPU__  void        free(void *ptr);
    //
    // sanity check, JTAG
    //
    __BOTH__ void        show_stat();
    __BOTH__ void        dump_freelist();

private:
    __GPU__  U32         _idx(U32 sz);
    __GPU__  S32         _find_free_index(U32 sz);
    __GPU__  void        _split(free_block *blk, U32 bsz);
    __GPU__  void        _pack(free_block *b0, free_block *b1);
    __GPU__  void        _unmap(free_block *blk);

    __GPU__  void        _mark_free(free_block *blk);
    __GPU__  free_block* _mark_used(U32 index);
    __GPU__  void        _try_merge_next(free_block *b0);
    __GPU__  free_block* _try_merge_prev(free_block *b1);

    /// mmu sanity check
    __BOTH__ int         _mmu_ok();
};
#endif // TEN4_SRC_TLSF_H_
