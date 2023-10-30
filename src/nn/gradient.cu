/** -*- c++ -*-
 * @file
 * @brief Model class - gradient descent functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
#include "dataset.h"

__KERN__ void k_sgd(
    DU *G, DU *DG, DU *M,                    ///< w, dw, and momemtum tensors
    int N, int numel,                        ///< batch size and HWC
    DU lr, DU b                              ///< learn rate, beta(momemtum)
    ) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;   ///< element index
    
    if (i < numel) {
        if (ABS(b) < DU_EPS) G[i] -= lr * DG[i] / N;
        else {
            DU dg = DG[i] / N;                             ///< dG batch avg
            DU mi = M[i] = b * M[i] + (1.0 - b) * dg;      ///< momentum
            G[i] -= lr * mi;                               /// * update gradient
        }
        DG[i] = DU0;                                       /// * zero after batch
    }
}

__KERN__ void k_adam(
    DU *G, DU *DG, DU *M, DU *V,            ///< w, dw, and momemtum tensors
    int N, int numel,                       ///< batch size and HWC
    DU lrc, DU b1, DU b2                    ///< corrected learn rate, beta(momemtum)
    ) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;   ///< element index
    
    if (i < numel) {
        DU dg = DG[i] / N;                                 ///< dG batch avg
        DU mi = M[i] = b1 * M[i] + (DU1 - b1) * dg;        ///< momentum
        DU vi = V[i] = b2 * V[i] + (DU1 - b2) * dg * dg;   ///< velocity
        G[i] -= lrc * mi / (SQRT(vi) + DU_EPS);            /// * update gradient
        DG[i] = DU0;                                       /// * zero out dG
    }
}

///
///> grad_alloc
///  @brief - allocate Momentum and Velocity tensors
///
__GPU__ Model&
Model::grad_alloc(t4_optimizer op) {
    for (int i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i];
        Tensor *w = in.grad[0], *dw = in.grad[2];
        Tensor *b = in.grad[1], *db = in.grad[3];

        bool do_w = dw && dw->is_same_shape(*w);  ///< exception: dropout
        bool do_b = db && db->is_same_shape(*b);  ///< exception: batchnorm
        
        switch (op) {
        case OPTI_SGD:
            in.mtum[0] = do_w ? w : NULL; in.mtum[2] = NULL;  /// * dummy
            in.mtum[1] = do_b ? b : NULL; in.mtum[3] = NULL;
            break;
        case OPTI_SGDM:
            if (do_w && !in.mtum[0]) {
                in.mtum[0] = &_mmu->copy(*dw).fill(DU0); ///< m of w (zero filled)
                in.mtum[2] = NULL;                       ///< dummy
            }
            if (do_b && !in.mtum[1]) {
                in.mtum[1] = &_mmu->copy(*db).fill(DU0); ///< m of b (zero filled)
                in.mtum[3] = NULL;                       ///< dummy
            }
            break;
        case OPTI_ADAM:
            if (do_w && !in.mtum[0]) {
                in.mtum[0] = &_mmu->copy(*dw).fill(DU0); ///< m of w (zeor filled)
                in.mtum[2] = &_mmu->copy(*dw).fill(DU0); ///< v of w (zero filled)
            }
            if (do_b && !in.mtum[1]) {
                in.mtum[1] = &_mmu->copy(*db).fill(DU0); ///< m of b (zero filled)
                in.mtum[3] = &_mmu->copy(*db).fill(DU0); ///< v of b (zero filled)
            }
            break;
        }
        TRACE1("Model::grad_alloc %2d> %s do_w,b[%d,%d] mtum=%p,%p,%p,%p\n",
            i, d_nname(in.grad_fn), do_w, do_b,
            in.mtum[0], in.mtum[1], in.mtum[2], in.mtum[3]);
    }
    return *this;
}
///
///> grandiant descent iterator
///
__GPU__ Model&
Model::gradient(const char *nm, GdFunc fn, DU *parm, t4_optimizer op) {
    auto step = [this, fn, parm](const char n,
            Tensor *g, Tensor *dg, Tensor *m, Tensor *v) {
            TRACE1("\n    %c[%d,%d,%d,%d] Σ=%6.3f - %6.3f",
                   n, g->N(), g->H(), g->W(), g->C(), g->sum(), dg->sum());
            fn(parm, g, dg, m, v);
            TRACE1(" => %cΣ=%6.3f", n, g->sum());
    };
    TRACE1("\nModel::%s batch_sz=%d, lr=%7.4f, mtum/b1=%6.3f, b2=%6.3f\n",
           nm, (*this)[1].N(), parm[0], parm[1], parm[2]);
    if (_iter++==0) grad_alloc(op);               /// * allocate m & v tensors
    if (!train) return *this;                     /// * bail if not in trainning
    ///
    /// cascade execution layer by layer forward
    ///
    DU t0 = _mmu->ms();                           ///< performance measurement
    for (U16 i = 1; i < numel - 1; i++) {         /// TODO: parallel layer update
        Tensor &in = (*this)[i];
        Tensor *w  = in.grad[0], *dw = in.grad[2];
        Tensor *b  = in.grad[1], *db = in.grad[3];
        
        TRACE1("\n  %2d> %s", i, d_nname(in.grad_fn));
        if (in.mtum[0]) step('w', w, dw, in.mtum[0], in.mtum[2]);
        if (in.mtum[1]) step('b', b, db, in.mtum[1], in.mtum[3]);
    }
    TRACE1("\nModel::%s %5.2f ms\n", nm, _mmu->ms() - t0);
    return *this;
}
///
/// Stochastic Gradient Descent
/// Note: does not get affected by batch size
///       because filters are fixed size
///
__GPU__ Model&
Model::sgd(DU lr, DU b) {                          /// a=momentum
    auto update = [](DU *parm, Tensor *g, Tensor *dg, Tensor *m, Tensor *v) {
        const int numel = g->numel;
        const dim3 blk(T4_WARP_SQ, 1, 1);          ///< default blocks
        const dim3 grd((numel + blk.x - 1) / blk.x, 1, 1);

        k_sgd<<<grd,blk>>>(
            g->data, dg->data, m->data,
            g->N(), numel, parm[0], parm[1]);
        GPU_SYNC();
    };
    DU parm[2] = { lr, _iter ? b : DU0 };

    return gradient("sgd", update, parm, ABS(b) < DU_EPS ? OPTI_SGD : OPTI_SGDM);
}

__GPU__ Model&
Model::adam(DU lr, DU b1, DU b2) {
    auto update = [](DU *parm, Tensor *g, Tensor *dg, Tensor *m, Tensor *v) {
        const int numel = g->numel;
        const dim3 blk(T4_WARP_SQ, 1, 1);         ///< default blocks
        const dim3 grd((numel + blk.x - 1) / blk.x, 1, 1);

        k_adam<<<grd,blk>>>(
            g->data, dg->data, m->data, v->data,
            g->N(), numel, parm[0], parm[1], parm[2]);
        GPU_SYNC();
    };
    DU parm[3] = {
        lr * SQRT(DU1 - POW(b2, _iter+1)) / (DU1 - POW(b1, _iter+1)),
        b1, b2                                    /// * corrected learn rate, betas
        // epoch ? b1 : DU0, epoch ? b2 : DU0     /// ** adjusted init b1, b2
    };
    return gradient("adam", update, parm, OPTI_ADAM);
}
#endif  // (T4_ENABLE_OBJ && T4_ENABLE_NN)
//==========================================================================
