/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider host implemenation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <sstream>
#include <string>
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "mnist.h"

namespace t4::ld {
///
/// for init debug MAX_BATCH 3
///
//#define MAX_BATCH 3          /**< debug, limit number of mini-batches */
#define MAX_BATCH 0          /**< debug, limit number of mini-batches */

Corpus *Mnist::init(bool trace) {
    auto _u32 = [this](std::ifstream &fs) {
        U32 v = 0;
        char x;
        for (int i = 0; i < 4; i++) { /// Big-Endian
            fs.read(&x, 1);
            v <<= 8;
            v += (U32)*(U8*)&x;
        }
        return v;
    };
    if (_open()) return NULL;
    data  = NULL;
    label = NULL;
    min   = 0;
    max   = 256;

    U32 X0, X1, N1=0;
    if (_tg) {
        X1 = _u32(_tg);    ///< label magic number 0x0801
        N1 = _u32(_tg);
        if (trace) INFO("\tMNIST label: magic=%08x => [%d]\n", X1, N1);
    }
    if (_ds) {
        X0 = _u32(_ds);    ///< image magic number 0x0803
        N  = _u32(_ds);
        H  = _u32(_ds);
        W  = _u32(_ds);
        C  = 1;
        if (trace) INFO("\tMNIST image: magic=%08x => [%d][%d,%d,%d]\n",
              X0, N, H, W, C);
    }
    if (N != N1) {
        ERROR("Mnist::init label count %d != image count %d\n", N1, N);
        return NULL;
    }
    return this;
}

Corpus *Mnist::fetch(int bid, int n, bool trace) {
    int off = n * bid;                          ///< batch offset index
    if (eof || off >= N) {                      /// * beyond total sample count?
        ERROR("Mnist::fetch EOF reached (needs rewind)\n");
        eof=1; return this;
    }
    ///
    /// fetch labels and images (and set eof if any of EOF reached)
    ///
    int b0   = _get_labels(bid, sizeof(U32) * 2, n * sizeof(U8));  ///< load batch labels
    batch_sz = _get_images(bid, sizeof(U32) * 4, n * cell());      ///< load batch images
    GPU_CHK();                                  /// * device sync after memory update
    
    if (b0 != batch_sz) {
        ERROR("Mnist::fetch #label=%d != #image=%d\n", b0, batch_sz);
        return NULL;
    }
    if ((off += batch_sz) >= N) eof = 1;        /// * EOF reached
    ///
    /// control partial batch for debugging
    ///
    if (MAX_BATCH && ((bid+1) >= MAX_BATCH)) {  /// * forced stop? (debug)
        batch_sz = n >> 1;                      /// * fake a partial batch
        eof = 1;
    }
    if (trace) {
        INFO("\tMnist batch[%d] loaded=%d/%d done=%d\n", bid, off, N, eof);
    }
    return this;
}

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
