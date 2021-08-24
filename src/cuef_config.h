/*! @file
  @brief
  cueForth - global configration for VM's
*/
#ifndef CUEF_SRC_CUEF_CONFIG_H_
#define CUEF_SRC_CUEF_CONFIG_H_

/* debugging flags
 *   cuef: general debugging controlled by -t option, default 1
 *   mmu : MMU alloc/free TLSF dump, default 0
 *   cc  : for my development only, default 0
 */
#define CUEF_DEBUG		    1
#define MMU_DEBUG			0
#define CC_DEBUG			0

/* min, maximum number of VMs */
#define MIN_VM_COUNT 		2
#define MAX_VM_COUNT 		16

/* maximum size of exception stack and registers, which determine how deep call stack can go */
#define VM_RESCUEF_STACK 	8
#define VM_REGFILE_SIZE 	128

/* max objects in symbol and global/constant caches allowed */
#define MAX_CONST_COUNT 	64
#define MAX_SYMBOL_COUNT 	256

/* cueForth can minimize usage for micro device */
#define CUEF_USE_STRING 	1
#define CUEF_USE_CONSOLE	0
#define CUEF_USE_FLOAT  	1
#define CUEF_USE_ARRAY  	1
#define CUEF_USE_MATH   	0
/*
 * 32it alignment is required
 * 	  0: Byte alignment
 * 	  1: 32bit alignment
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
#define CUEF_32BIT_ALIGN_REQUIRED 	1
#define CUEF_HEAP_SIZE 				(64*1024)
#define CUEF_STRBUF_SIZE			(256-1)
#define CUEF_OUTPUT_BUF_SIZE        4096

/* CUDA dependent flags */
#define CUDA_MIN_MEMBLOCK_SIZE		0x200
#define CUDA_ENABLE_CDP            	0

#endif // CUEF_SRC_CUEF_CONFIG_H_




