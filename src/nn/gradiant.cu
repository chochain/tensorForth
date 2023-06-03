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
            TRACE1("%c[%d,%d,%d,%d] Σ=%6.3f - %6.3f",
                   n, g.N(), g.H(), g.W(), g.C(), g.sum(), dg.sum());
            fn(g, dg, _gparm, _gzero);
            TRACE1(" => %cΣ=%6.3f", n, g.sum());
    };
    Tensor &n1 = (*this)[1];                       ///< reference model input layer
    DU     t0  = _mmu->ms();                       ///< performance measurement
    ///
    /// cascade execution layer by layer forward
    ///
    const int N = n1.N();                          ///< batch size
    TRACE1("\nModel#%s batch_sz=%d, lr=%6.3f, mtum/b1=%6.3f b2=%6.3f",
           nm, N, _gparm[0], _gparm[1], _gparm[2]);
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i];
        Tensor *w  = in.grad[0], *dw = in.grad[2];
        Tensor *b  = in.grad[1], *db = in.grad[3];
        
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
        const int N  = g.N();
        const DU  lr = parm[0];
        const DU  m  = parm[1];
        if (m < DU_EPS) {
            dg *= lr / N;                          /// * learn rate / batch size
            g  -= dg;                              /// * g -= eta * dg
        }
        else {                                     /// * with momentum (exp moving avg)
            dg *= (1 - m) * lr / N;                /// * w' = m * w - (1 - m) * eta * dw
            g  *= m;
            g  -= dg;
        }
        if (zero) dg.map(O_FILL, DU0);             /// * zap dw, ready for next batch
    };
    _gparm[0] = zero ? DU1 : DU0;
    _gparm[1] = lr;
    _gparm[2] = m;
    _gparm[3] = DU0;
    gradiant("sgd", update);
    return *this;
}

__GPU__ Model&
Model::adam(DU lr, DU b1, DU b2, bool zero) {
    auto update = [](Tensor &g, Tensor &dg, DU *parm, bool zero) {
        const int N  = g.N();
        const DU  lr = parm[0];
        const DU  b1 = parm[1];
        const DU  b2 = parm[2];
        Tensor &v = Tensor::copy(dg);
        Tensor::matmul(dg, v, v);      /// * v = dw^2
        /*
        dw *= (1 - m) * lr / N;                /// * w' = m * w - (1 - m) * eta * dw
        w  *= m;
        w  -= dw;
        */
        _mmu.free(v);
    };
    _zero     = zero;
    _gparm[0] = lr;
    _gparm[1] = b1;
    _gparm[2] = b2;
    gradiant("adam", update);
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
