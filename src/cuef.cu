/*! @file
  @brief
  cueForth value definitions non-optimized
*/
#include <stdio.h>
#include "cuef.h"

// forward declaration for implementation
extern "C" __GPU__  void cuef_mmu_init(void *ptr, U32 sz);
extern "C" __GPU__  void cuef_core_init(void);
extern "C" __GPU__  void cuef_console_init(U8 *buf, U32 sz);

CuefSes *_ses_list = NULL; 	  // session linked-list

__HOST__ int
cuef_setup(int step, int trace)
{
	cudaDeviceReset();

	debug_init(trace);												// initialize logger
	debug_log("cueF initializing...");

	U8 *mem = cuef_host_heap = (U8*)cuda_malloc(CUEF_HEAP_SIZE, 1);	// allocate main block (i.e. RAM)
	if (!mem) 				return -11;
	U8 *out = _cuef_out = (U8*)cuda_malloc(OUTPUT_BUF_SIZE, 1);		// allocate output buffer
	if (!_cuef_out) 		return -12;
	if (vm_pool_init(step)) return -13;								// allocate VM pool

	_ses_list = NULL;

	cuef_mmu_init<<<1,1>>>(mem, CUEF_HEAP_SIZE);			// setup memory management
	cuef_core_init<<<1,1>>>();								// setup basic classes	(TODO: => ROM)
#if CUEF_USE_CONSOLE
	cuef_console_init<<<1,1>>>(out, OUTPUT_BUF_SIZE);		// initialize output buffer
#endif
	GPU_SYNC();

    U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	debug_log("cuef initialized, ready to go...");

	return 0;
}

__HOST__ int
cuef_load(char *rite_name)
{
	CuefSession ses = new CuefSession(cin, cout);

	char *ins = _fetch_bytecode(rite_name);
	if (!ins) return -22;		// bytecode memory allocation error

	int id = ses->id = vm_get(ins);
	cuda_free(ins);

	if (id<0) return id;

	ses->next = _ses_list;		// add to linked-list
	_ses_list = ses;

	return 0;
}

__HOST__ int
cuef_run()
{
	debug_log("cuef session starting...");
	debug_mmu_stat();

	// parse BITE code into each vm
	// TODO: work producer (enqueue)
	for (cuef_ses *ses=_ses_list; ses!=NULL; ses=ses->next) {
		int id = vm_ready(ses->id);
		if (id) return id;
	}
	// kick up main loop until all VM are done
	vm_main_start();

	debug_mmu_stat();
	debug_log("cuef session completed.");

	return 0;
}

__HOST__ void
cuef_teardown(int sig)
{
	cudaDeviceReset();

	cuef_ses *tmp, *ses = _ses_list;
	while (ses) {
		tmp = ses;
		ses = ses->next;
		free(tmp);
	}
}
