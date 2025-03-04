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
        printf("\n\tdf[%d]=", n);
        for (U32 i = 0; i < df.HWC(); i++) {
            sum  += *v;
            fsum += *v;
            printf("%5.2f ", *v++);
        }
        printf("Σ=%5.2f", fsum);
    }
    printf("\n\tΣΣ=%6.3f", sum);
}

__GPU__ void
Model::_dump_db(Tensor &db) {
    DU sum = DU0;
    printf("\n\tdb=");
    DU *v = db.data;
    for (U32 i = 0; i < db.H(); i++) {
        printf("%6.3f ", *v);
        sum += *v++;
    }
    printf(" Σ=%6.3f", sum);
}

__GPU__ void
Model::_dump_dw(Tensor &dw, bool full) {
    const U32 = dw.H(), W = dw.W();
    DU hsum = DU0, *p = dw.data;
    if (full) printf("\ndw[%d,%d]=", W, H);
    else      printf("\n\tdwΣ=");
    for (U32 i = 0; i < H; i++) {
        if (full) printf("\n\tdw[%d]=", i);
        DU sum = DU0;
        for (U32 j = 0; j < W; j++, p++) {
            sum  += *p;
            hsum += *p;
            if (full) printf("%6.3f", *p);
        }
        if (full) printf(" Σ=%6.3f", sum);
        else      printf("%6.3f ", sum);
    }
    if (H > 1) printf("%sΣΣ=%6.3f", full ? "\n\t" : " ", hsum);
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
