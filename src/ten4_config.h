/*! @file
  @brief
  tensorForth - global configration for VM's
*/
#ifndef TEN4_SRC_TEN4_CONFIG_H_
#define TEN4_SRC_TEN4_CONFIG_H_

/* debugging flags
 *   t4  : general debugging controlled by -t option, default 1
 *   mmu : MMU alloc/free TLSF dump, default 0
 *   cc  : for my development only, default 0
 */
#define T4_APP_NAME        "tensorForth"
#define T4_DEBUG            0
#define MMU_DEBUG           0
#define CC_DEBUG            0

/* tensorForth can minimize usage for micro device */
#define T4_ENABLE_CDP       0
#define T4_USE_STRBUF       0


/* min, maximum number of VMs */
#define MIN_VM_COUNT        1
#define MAX_VM_COUNT        16

/* maximum size of exception stack and registers, which determine how deep call stack can go */
#define VM_RESTEN4_STACK    8
#define VM_REGFILE_SZ       128

/* max objects in symbol and global/constant caches allowed */
#define MAX_CONST_COUNT     64
#define MAX_SYMBOL_COUNT    256
/*
 * 32it alignment is required
 *    0: Byte alignment
 *    1: 32bit alignment
 * Heap size
 *    48K can barely fit utf8_02 test case
 * string buffer size
 *    for strcat and puts
 * host grit image
 *    0 : GRIT created inside GPU
 *    1 :              by CPU
 * CXX codebase
 *    0 : use C only
 *    1 : use C++ set
 */
#define T4_32BIT_ALIGN_REQUIRED   1
#define T4_PMEM_SZ                (48*1024) /** parameter memory block size  */
#define T4_RS_SZ                  64        /** depth of return stack        */
#define T4_SS_SZ                  64        /** depth of data stack          */
#define T4_DICT_SZ                1024      /** number of dictionary entries */
#define T4_IBUF_SZ                1024      /** host input buffer size       */
#define T4_OBUF_SZ                8192      /** device output buffer size    */
#define T4_STRBUF_SZ              128       /** temp string buffer size      */

#endif // TEN4_SRC_TEN4_CONFIG_H_
