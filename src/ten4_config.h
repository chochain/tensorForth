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
#define T4_APP_NAME         "tensorForth v4.0"
#define T4_VERBOSE          1        /**< system verbose 0|1|2  */
#define T4_CASE_SENSITIVE   1        /**< interpreter case      */
#define MM_DEBUG            1        /**< for my local testing  */
///@}
///@name CUDA cooperative dynamic parallelism support
///@{
#define T4_ENABLE_OBJ       1        /**< enable tensor/matrix  */
#define T4_ENABLE_NN        0        /**< enable neural network */
#define T4_ENABLE_CDP       0
#define T4_USE_STRBUF       0
#define T4_PER_THREAD_STACK 8*1024   /**< init() stack overflow */
///@}
///@name Virtual machine instance controls
///@{
#define T4_VM_COUNT         4        /**< number of VMs         */
#define T4_EXP_STACK        8        /**< exception stack depth */
#define T4_REGFILE_SZ       128      /**< register file size    */
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
#define T4_ALIGN4    1         /**< data 32-bit aligned          */
#define T4_PMEM_SZ   (48*1024) /**< parameter memory block size  */
#define T4_USER_AREA (ALIGN16(T4_VM_COUNT))
#define T4_RS_SZ     64        /**< depth of return stack        */
#define T4_SS_SZ     64        /**< depth of data stack          */
#define T4_NET_SZ    32        /**< size of network DAG          */
#define T4_DICT_SZ   1024      /**< number of dictionary entries */
#define T4_IBUF_SZ   1024      /**< host input buffer size       */
#define T4_OBUF_SZ   8192      /**< device output buffer size    */
#define T4_STRBUF_SZ 128       /**< temp string buffer size      */
#define T4_OSTORE_SZ (1024*1024*1024) /**< object storage size   */ 
#define T4_TFREE_SZ  T4_NET_SZ /**< size of tensor free queue    */
#define T4_RAND_SZ   256       /**< number of random seeds       */
#define T4_WARP_SZ   16        /**< CUDA GPU warp 16x16 threads  */
#define T4_WARP_SQ   (T4_WARP_SZ * T4_WARP_SZ)
///@}
#endif // TEN4_SRC_TEN4_CONFIG_H_
