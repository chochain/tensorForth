/*!
 * @file
 * @brief - tensorForth TLSF unit tests
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>          // cin, cout
#include <signal.h>
using namespace std;

#include "../src/ten4_config.h"
#include "../src/ten4_types.h"
#include "../src/tlsf.h"

__HOST__ void
test_host_init(U8 *data, U32 sz) {
    printf("test HOST init ====================\n");
    TLSF tlsf;
    tlsf.init(data, sz);
    tlsf.show_stat();
    tlsf.dump_freelist();
}

__KERN__ void
test_kern_init(U8 *data, U32 sz) {
    if (blockIdx.x !=0 || threadIdx.x != 0) return;

    printf("test KERN init =====================\n");
    TLSF tlsf;
    tlsf.init(data, sz);
    tlsf.show_stat();
    tlsf.dump_freelist();
}

__KERN__ void
test_alloc(U8 *data, U32 sz) {
    if (blockIdx.x !=0 || threadIdx.x != 0) return;

    U32 a[4] = { 1024 * 2048, 2048 * 512, 1024 * 512, 1024 * 512 };
    DU  *blk[4];
    TLSF tlsf;
    tlsf.init(data, sz);
    printf("test KERN malloc ==============\n");
    for (int i=0; i<4; i++) {
        blk[i] = (DU*)tlsf.malloc(a[i]);
        printf("malloc(%x) => %p\n", a[i], blk[i]);
        tlsf.show_stat();
        tlsf.dump_freelist();
    }
    printf("test KERN free ==================\n");
    for (int i=0; i<4; i++) {
        printf("free(%p)\n", blk[i]);
        tlsf.free(blk[i]);
        tlsf.show_stat();
        tlsf.dump_freelist();
    }
    printf("test KERN 2nd malloc ====================\n");
    for (int i=0; i<4; i++) {
        blk[i] = (DU*)tlsf.malloc(a[i]);
        printf("malloc(%x) => %p\n", a[i], blk[i]);
        tlsf.show_stat();
        tlsf.dump_freelist();
    }
    printf("test KERN 2nd free ===============\n");
    for (int i=3; i>=0; i--) {
        printf("free(%p)\n", blk[i]);
        tlsf.free(blk[i]);
        tlsf.show_stat();
        tlsf.dump_freelist();
    }
}

int main(int argc, char**argv) {
    printf("%s tests start ===============\n", argv[0]);
    U8 *data;
    U32 sz = T4_TENSOR_SZ;
    cudaMallocManaged((void**)&data, sz);
    GPU_CHK();

    test_host_init(data, sz);
    test_kern_init<<<1,1>>>(data, sz);
    GPU_CHK();

    test_alloc<<<1,1>>>(data, sz);
    GPU_CHK();

    cudaFree(data);
    GPU_CHK();

    printf("%s tests done ===============\n", argv[0]);
    return 0;
}
