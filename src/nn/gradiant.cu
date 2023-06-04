/** -*- c++ -*-
 * @file
 * @brief Model class - gradiant descent functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
__GPU__ Model&
Model::gradiant(const char *nm, GdFunc fn) {
    auto step = [this, fn](const char n, Tensor &g, Tensor &dg) {
            TRACE1("\n    %c[%d,%d,%d,%d] Σ=%6.3f - %6.3f",
                   n, g.N(), g.H(), g.W(), g.C(), g.sum(), dg.sum());
            fn(g, dg, _gparm, _gzero);
            TRACE1(" => %cΣ=%6.3f", n, g.sum());
    };
    Tensor &n1 = (*this)[1];                       ///< reference model input layer
    DU     t0  = _mmu->ms();                       ///< performance measurement
    ///
    /// cascade execution layer by layer forward
    ///
    TRACE1("\nModel#%s batch_sz=%d, lr=%6.3f, mtum/b1=%6.3f b2=%6.3f",
           nm, n1.N(), _gparm[0], _gparm[1], _gparm[2]);
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i];
        Tensor *w  = in.grad[0], *dw = in.grad[2];
        Tensor *b  = in.grad[1], *db = in.grad[3];
        
        if (_trace) printf("\n  %2d> %s", i, d_nname(in.grad_fn));
        if (dw && dw->is_same_shape(*w)) step('w', *w, *dw);
        if (db && db->is_same_shape(*b)) step('b', *b, *db);
    }
    TRACE1("\nModel#%s %5.2f ms\n", nm, _mmu->ms() - t0);
    return *this;
}
///
/// Stochastic Gradiant Descent
/// Note: does not get affected by batch size
///       because filters are fixed size
///
__GPU__ Model&
Model::sgd(DU lr, DU m, bool zero) {
    auto update = [](Tensor &g, Tensor &dg, DU *parm, bool zero) {
        const DU m = parm[1];                      ///< momentum
        if (m < DU_EPS) {
            dg *= parm[0];                         /// * eta / batch size
            g  -= dg;                              /// * g -= eta * dg
        }
        else {                                     /// * with momentum (exp moving avg)
            dg *= (1 - m) * parm[0];               /// * w' = m * w - (1 - m) * eta * dw / N
            g  *= m;
            g  -= dg;
        }
        if (zero) dg.map(O_FILL, DU0);             /// * zap dw, ready for next batch
    };
    _gparm[0] = lr / batch_size();                 /// eta / batch_size
    _gparm[1] = m;
    _gparm[2] = DU0;
    _gzero    = zero;
    gradiant("sgd", update);
    return *this;
}

__GPU__ Model&
Model::adam(DU lr, DU b1, DU b2, bool zero) {
    static DU t = 1;
    auto update = [](Tensor &g, Tensor &dg, DU *parm, bool zero) {
        const DU  lr = parm[0];                   /// * eta / batch_size
        const DU  b1 = parm[1];
        const DU  b2 = parm[2];
    };
    _gparm[0] = lr * SQRT(1 - POW(b2, t)) / (1 - POW(b1, t)) / batch_size();
    _gparm[1] = b1;
    _gparm[2] = b2;
    _gzero    = zero;
    gradiant("adam", update);
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
