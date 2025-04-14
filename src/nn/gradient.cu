/** -*- c++ -*-
 * @file
 * @brief Model class - gradient descent functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "dataset.h"

__KERN__ void k_sgd(
    DU *G, DU *DG, DU *M,                   ///< w, dw, and momemtum tensors
    U32 N, DU lr, DU b,                     ///< batch size, learn rate, beta(momemtum)
    U64 numel                               ///< HWC
    ) {
    for (U64 j = threadIdx.x; j < numel; j += blockDim.x) {
        if (ABS(b) < DU_EPS) G[j] -= lr * DG[j] / N;
        else {
            DU dg = DG[j] / N;                             ///< dG batch avg
            DU mi = M[j] = b * M[j] + (1.0 - b) * dg;      ///< momentum
            G[j] -= lr * mi;                               /// * update gradient
        }
        DG[j] = DU0;                                       /// * zero after batch
    }
}

__KERN__ void k_adam(
    DU *G, DU *DG, DU *M, DU *V,            ///< w, dw, and momemtum tensors
    U32 N, DU lrc, DU b1, DU b2,            ///< batch size,corrected learn rate, beta(momemtum)
    U64 numel                               ///< HWC
    ) {
    for (U64 j = threadIdx.x; j < numel; j += blockDim.x) {
        const DU dg = DG[j];                                     ///< dG (no batch avg)
        const DU mi = M[j] = b1 * M[j] + (DU1 - b1) * dg;        ///< momentum
        const DU vi = V[j] = b2 * V[j] + (DU1 - b2) * dg * dg;   ///< velocity
        
        G[j] -= lrc * mi / (SQRT(vi) + DU_EPS);                  /// * update gradient, clipped
        DG[j] = DU0;                                             /// * zero out dG for next round
    }
}

///
///> grad_alloc
///  @brief - allocate Momentum and Velocity tensors
///
#define M2X(i)     (in.mtum[i] ? _mmu->OBJ2X(*in.mtum[i]) : 0)
__GPU__ Model&
Model::grad_alloc(t4_optimizer op) {
    NN_DB("  #grad_alloc {\n");
    for (int i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i];
        Tensor *w = in.grad[0], *dw = in.grad[2];   ///< filter tensor pointers
        Tensor *b = in.grad[1], *db = in.grad[3];   ///< bias tensor pointers

        bool do_w = dw && dw->is_same_shape(*w);    ///< exception: dropout
        bool do_b = db && db->is_same_shape(*b);    ///< exception: batchnorm
        
        switch (op) {
        case OPTI_SGD:
            in.mtum[0] = do_w ? w : NULL; in.mtum[2] = NULL;  /// * dummy
            in.mtum[1] = do_b ? b : NULL; in.mtum[3] = NULL;
            break;
        case OPTI_SGDM:
            if (do_w && !in.mtum[0]) {
                in.mtum[0] = &COPY(*dw).fill(DU0);  ///< m of w (zero filled)
                in.mtum[2] = NULL;                  ///< dummy
            }
            if (do_b && !in.mtum[1]) {
                in.mtum[1] = &COPY(*db).fill(DU0);  ///< m of b (zero filled)
                in.mtum[3] = NULL;                  ///< dummy
            }
            break;
        case OPTI_ADAM:
            if (do_w && !in.mtum[0]) {
                in.mtum[0] = &COPY(*dw).fill(DU0);  ///< m of w (zeor filled)
                in.mtum[2] = &COPY(*dw).fill(DU0);  ///< v of w (zero filled)
            }
            if (do_b && !in.mtum[1]) {
                in.mtum[1] = &COPY(*db).fill(DU0);  ///< m of b (zero filled)
                in.mtum[3] = &COPY(*db).fill(DU0);  ///< v of b (zero filled)
            }
            break;
        }
        NN_DB("    %d> %s do_w,b[%d,%d] mtum=%x,%x,%x,%x\n",
              i, d_nname(in.grad_fn), do_w, do_b,
              M2X(0), M2X(1), M2X(2), M2X(3));
    }
    NN_DB("  } #grad_alloc\n");
    return *this;
}
///
///> grandiant descent iterator
///
__GPU__ Model&
Model::gradient(const char *nm, t4_optimizer op, GdFunc fn, DU *parm) {
    auto step = [this, fn, parm](const char k,
        Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        NN_DB("     %c[%2d,%2d,%2d,%2d] Σ=%6.3f - %6.3f",
              k, g.N(), g.H(), g.W(), g.C(), g.sum(), dg.sum());
#if MM_DEBUG        
        Tensor::_dump(g.data, g.H(), g.W(), g.C());
        Tensor::_dump(dg.data, dg.H(), dg.W(), dg.C());
        fn(parm, g, dg, m, v);                /// * execute grad function
        Tensor::_dump(g.data, g.H(), g.W(), g.C());
        Tensor::_dump(dg.data, dg.H(), dg.W(), dg.C());
#else  // !MM_DEBUG
        fn(parm, g, dg, m, v);                /// * execute grad function
#endif // MM_DEBUG
        DU sum0 = g.sum();
        NN_DB(" => %cΣ=%6.3f", k, sum0);
        if (max_norm > DU_EPS) {
            DU std = g.std();
            if (std > max_norm) {
                g *= max_norm / std;
                NN_DB(" std=%6.3% => adj. %cΣ=%6.3f");
            }
        }
        NN_DB("\n");
    };
    NLOG("\nModel::%s starts batch_sz=%d, lr=%7.4f, mtum/b1=%6.3f, b2=%6.3f max_norm=%6.3f {\n",
         nm, (*this)[1].N(), parm[0], parm[1], parm[2], max_norm);
    if (epoch==0 && _iter++==0) grad_alloc(op);   /// * allocate m & v tensors
    if (!train) return *this;                     /// * bail if not in trainning
    ///
    /// cascade execution layer by layer forward
    ///
    DU t0 = System::ms();                         ///< performance measurement
    for (int i = 1; i < numel - 1; i++) {         /// TODO: parallel layer update
        Tensor &in = (*this)[i];
        Tensor &w  = *in.grad[0], &dw = *in.grad[2];
        Tensor &b  = *in.grad[1], &db = *in.grad[3];

        if (*_trace) INFO("  %d> %s\n", i, d_nname(in.grad_fn));
        
        if (in.mtum[0]) step('w', w, dw, *in.mtum[0], *in.mtum[2]);
        if (in.mtum[1]) step('b', b, db, *in.mtum[1], *in.mtum[3]);
    }
    NLOG("} Model::%s %5.2f ms\n", nm, System::ms() - t0);
    return *this;
}
///
/// Stochastic Gradient Descent
/// Note: does not get affected by batch size
///       because filters are fixed size
///
__GPU__ Model&
Model::sgd(DU lr, DU b) {                          /// b=beta (momentum)
    auto update = [](DU parm[4], Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        FORK1(k_sgd, g.numel, 
             g.data, dg.data, m.data,
             g.N(), parm[1], parm[2]);
        CDP_SYNC();
    };
    DU parm[3] = { lr, epoch ? b : DU0, DU0 };

    return gradient("sgd", ABS(b) < DU_EPS ? OPTI_SGDM : OPTI_SGDM, update, parm);
}

__GPU__ Model&
Model::adam(DU lr, DU b1, DU b2) {
    auto update = [](DU *parm, Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        FORK1(k_adam, g.numel,
             g.data, dg.data, m.data, v.data,
             g.N(), parm[0], parm[1], parm[2]);
        CDP_SYNC();
    };
    DU parm[3] = {                    ///< learn rate, betas
        lr * SQRT(DU1 - POW(b2, epoch+1)) / (DU1 - POW(b1, epoch+1)),
        b1, b2
    };
    return gradient("adam", OPTI_ADAM, update, parm);
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
