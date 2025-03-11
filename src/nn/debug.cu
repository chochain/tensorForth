/** -*- c++ -*-
 * @file
 * @brief Model class - debug/trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "dataset.h"

__GPU__ void
Model::_dump_dbdf(Tensor &db, Tensor &df) {
    _dump_db(db);
    
    DU sum = DU0;
    for (U32 n = 0; n < df.N(); n++) {
        DU *v = df.slice(n), fsum = DU0;
        INFO("\n\tdf[%d]=", n);
        for (U32 i = 0; i < df.HWC(); i++) {
            sum  += *v;
            fsum += *v;
            INFO("%5.2f ", *v++);
        }
        INFO("Σ=%5.2f", fsum);
    }
    INFO("\n\tΣΣ=%6.3f", sum);
}

__GPU__ void
Model::_dump_db(Tensor &db) {
    DU sum = DU0;
    INFO("\n\tdb=");
    DU *v = db.data;
    for (U32 i = 0; i < db.H(); i++) {
        INFO("%6.3f ", *v);
        sum += *v++;
    }
    INFO(" Σ=%6.3f", sum);
}

__GPU__ void
Model::_dump_dw(Tensor &dw, bool full) {
    const U32 H = dw.H(), W = dw.W();
    DU hsum = DU0, *p = dw.data;
    if (full) INFO("\ndw[%d,%d]=", W, H);
    else      INFO("\n\tdwΣ=");
    for (U32 i = 0; i < H; i++) {
        if (full) INFO("\n\tdw[%d]=", i);
        DU sum = DU0;
        for (U32 j = 0; j < W; j++, p++) {
            sum  += *p;
            hsum += *p;
            if (full) INFO("%6.3f", *p);
        }
        if (full) INFO(" Σ=%6.3f", sum);
        else      INFO("%6.3f ", sum);
    }
    if (H > 1) INFO("%sΣΣ=%6.3f", full ? "\n\t" : " ", hsum);
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
