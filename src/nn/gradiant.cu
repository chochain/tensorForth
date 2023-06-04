/** -*- c++ -*-
 * @file
 * @brief Model class - gradiant descent functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
__KERN__ void k_sgd(
    DU *G, DU *DG,                                  ///< func, input, output tensors
    DU lr, DU a, bool zero,
    int HW
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index
    
    if (k < HW) {
        G[k] = (a < DU_EPS)
            ? G[k] - lr * DG[k]
            : a * G[k] + (1 - a) * DG[k];     /// * with momentum (exp moving avg)
        if (zero) DG[k] = DU0;
    }
}

__KERN__ void k_adam(
    DU *G, DU *DG, DU *M, DU *V,     ///< func, input, output tensors
    DU lr, DU b1, DU b2, bool zero,  ///< lr = eta * sqrt(1 - b2^t) / (1 - b1^t) / N
    int HW
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index

    if (i < HW) {
        DU dg = DG[k];
        DU mk = M[k] = b1 * M[k] + (1 - b1) * dg;
        DU vk = V[k] = b2 * V[k] + (1 - b2) * dg * dg;
        G[k] -= lr * M[k] / (SQRT(V[k]) + DU_EPS);
        
        if (zero) DG[k] = DU0;
    }
}
///
///> grandiant descent iterator
///
__GPU__ Model&
Model::gradiant(const char *nm, GdFunc fn, t4_optimizer opti) {
    auto step = [this, fn](const char n,
            Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
            TRACE1("\n    %c[%d,%d,%d,%d] Σ=%6.3f - %6.3f",
                   n, g.N(), g.H(), g.W(), g.C(), g.sum(), dg.sum());
            fn(_gparm, _gzero, g, dg, m, v);
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
        ///
        /// * initialize adam m and v tensors for each layer if needed
        ///
        switch (opti) {
        case OPTI_SGD:
            if (dw && !in.adam[0]) in.adam[0] = w; in.adam[1] = dw;
            if (dw && !in.adam[2]) in.adam[2] = b; in.adam[1] = db;
            break;
        case OPTI_ADAM:
            if (dw && !in.adam[0]) {
                in.adam[0] = &_mmu->copy(*w).fill(DU0);     ///< m of w
                in.adam[1] = &_mmu->copy(*dw).fill(DU0);    ///< v of w
            }
            if (db && !in.adam[2]) {
                in.adam[2] = &_mmu->copy(*b).fill(DU0);     ///< m of b
                in.adam[3] = &_mmu->copy(*db).fill(DU0);    ///< v of b
            }
        }
        if (dw && dw->is_same_shape(*w))
            step('w', *w, *dw, *in.adam[0], *in.adam[1]);
        if (db && db->is_same_shape(*b))
            step('b', *b, *db, *in.adam[2], *in.adam[3]);
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
Model::sgd(DU lr, DU a, bool zero) {               /// a=momentum
    auto update = [](DU *parm, bool zero,
                     Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        const int HW = g.H() * g.W();
        const dim3 blk(T4_WARP_SQ, 1, 1);          ///< default blocks
        const dim3 grd((HW + blk.x - 1)/blk.x, g.C(), g.N());
        
        k_sgd<<<grd,blk>>>(g.data, dg.data, parm[0], parm[1], zero, HW);
    };
    _gparm[0] = lr / batch_size();                 /// eta / batch_size
    _gparm[1] = a;
    _gparm[2] = DU0;
    _gzero    = zero;
    gradiant("sgd", update, OPTI_SGD);
    return *this;
}

__GPU__ Model&
Model::adam(DU lr, DU b1, DU b2, bool zero) {
    static int t = 1;
    auto update = [](DU *parm, bool zero,
                     Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        const int HW = g.H() * g.W();
        const dim3 blk(T4_WARP_SQ, 1, 1);               ///< default blocks
        const dim3 grd((HW + blk.x - 1)/blk.x, g.C(), g.N());

        k_adam<<<grd,blk>>>(g.data, dg.data, m.data, v.data,
                            parm[0], parm[1], parm[2], zero, HW);
    };
    _gparm[0] = lr * SQRT(1 - POW(b2, t)) / (1 - POW(b1, t)) / batch_size();
    _gparm[1] = b1;
    _gparm[2] = b2;
    _gzero    = zero;
    gradiant("adam", update, OPTI_ADAM);
    
    t++;
    
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
