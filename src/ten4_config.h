/**
 * @file
 * @brief tensorForth global configration for VM's
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TEN4_CONFIG_H_
#define TEN4_SRC_TEN4_CONFIG_H_

/* debugging flags
 *   t4  : general debugging controlled by -t option, default 1
 *   mmu : MMU alloc/free TLSF dump, default 0
 *   cc  : for my development only, default 0
 */
///
///@name Debugging macros
///@{
#define T4_APP_NAME        "tensorForth"
#define T4_VERBOSE          1        /**< system verbose print */
#define T4_MMU_DEBUG        0        /**< for mmu debug trace  */
#define CC_DEBUG            1        /**< for my local testing */
///@}
///@name CUDA cooperative dynamic parallelism support
///@{
#define T4_ENABLE_OBJ       1
#define T4_ENABLE_CDP       0
#define T4_USE_STRBUF       0
///@}
///@name Virtual machine instance controls
///@{
#define VM_MIN_COUNT        1
#define VM_MAX_COUNT        16
#define VM_EXP_STACK        8        /**< exception stack depth */
#define VM_REGFILE_SZ       128      /**< register file size    */
///@}
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
///
///@name Storage sizing
///@{
#define T4_32BIT_ALIGN_REQUIRED   1
#define T4_PMEM_SZ                (48*1024) /**< parameter memory block size  */
#define T4_RS_SZ                  64        /**< depth of return stack        */
#define T4_SS_SZ                  64        /**< depth of data stack          */
#define T4_DICT_SZ                1024      /**< number of dictionary entries */
#define T4_IBUF_SZ                1024      /**< host input buffer size       */
#define T4_OBUF_SZ                8192      /**< device output buffer size    */
#define T4_STRBUF_SZ              128       /**< temp string buffer size      */
#define T4_TENSOR_SZ              (1*1024*1024*1024) /**< tensor storage size */
#define T4_TFREE_SZ               16        /**< size of tensor free queue    */
///@}
#endif // TEN4_SRC_TEN4_CONFIG_H_
