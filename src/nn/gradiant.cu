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
    DU *G, DU *DG, DU *M,        ///< w, dw, and momemtum tensors
    DU lr, DU b, int HW          ///< learn rate, beta(momemtum), and dw zero flag
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index
    
    if (k < HW) {
        if (b < DU_EPS) G[k] -= lr * DG[k];
        else {
            DU mk = M[k] = b * M[k] + (1.0 - b) * DG[k];
            G[k] -= lr * mk;
        }
        DG[k] = DU0;                                       /// * zero after batch
    }
}

__KERN__ void k_adam(
    DU *G, DU *DG, DU *M, DU *V, ///< w, dw, and momemtum tensors
    DU lr, DU b1, DU b2, int HW  ///< learn rate, beta(momemtum), and dw zero flag
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index
    
    if (k < HW) {
        DU dg = DG[k] / gridDim.z;
        DU mk = M[k] = b1 * M[k] + (1.0 - b1) * dg;
        DU vk = V[k] = b2 * V[k] + (1.0 - b2) * dg * dg;
        G[k] -= lr * M[k] / (SQRT(V[k]) + DU_EPS);
        DG[k] = DU0;
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

        bool do_w = dw && dw->is_same_shape(*w);
        bool do_b = db && db->is_same_shape(*b);
        
        switch (op) {
        case OPTI_SGD:
            in.mtum[0] = in.mtum[2] = dw;         /// * dummy, no extra storage
            in.mtum[1] = in.mtum[3] = db;
            break;
        case OPTI_SGDM:
            if (do_w && !in.mtum[0]) {
                in.mtum[0] = &_mmu->copy(*w);     ///< m of w
                in.mtum[2] = dw;                  ///< dummy
            }
            if (do_b && !in.mtum[1]) {
                in.mtum[1] = &_mmu->copy(*b);     ///< m of b
                in.mtum[3] = db;                  ///< dummy
            }
            break;
        case OPTI_ADAM:
            if (do_w && !in.mtum[0]) {
                in.mtum[0] = &_mmu->copy(*w);     ///< m of w
                in.mtum[2] = &_mmu->copy(*dw);    ///< v of w
            }
            if (do_b && !in.mtum[1]) {
                in.mtum[1] = &_mmu->copy(*b);     ///< m of b
                in.mtum[3] = &_mmu->copy(*db);    ///< v of b
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
Model::gradiant(const char *nm, GdFunc fn, DU *parm, t4_optimizer op) {
    auto step = [this, fn, parm](const char n,
            Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
            TRACE1("\n    %c[%d,%d,%d,%d] Σ=%6.3f - %6.3f",
                   n, g.N(), g.H(), g.W(), g.C(), g.sum(), dg.sum());
            fn(parm, g, dg, m, v);
            TRACE1(" => %cΣ=%6.3f", n, g.sum());
    };
    TRACE1("\nModel#%s batch_sz=%d, lr=%6.3f, mtum/b1=%6.3f b2=%6.3f\n",
           nm, (*this)[1].N(), parm[0], parm[1], parm[2]);
    
    if (train && _iter++==0) grad_alloc(op);      ///< allocate m & v tensors
    ///
    /// cascade execution layer by layer forward
    ///
    DU t0  = _mmu->ms();                          ///< performance measurement
    for (U16 i = 1; i < numel - 1; i++) {         /// TODO: parallel update
        Tensor &in = (*this)[i];
        Tensor *w  = in.grad[0], *dw = in.grad[2];
        Tensor *b  = in.grad[1], *db = in.grad[3];
        
        TRACE1("\n  %2d> %s", i, d_nname(in.grad_fn));
        
        if (dw) step('w', *w, *dw, *in.mtum[0], *in.mtum[2]);
        if (db) step('b', *b, *db, *in.mtum[1], *in.mtum[3]);
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
Model::sgd(DU lr, DU b) {                          /// a=momentum
    auto update = [](DU *parm, Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        const int HW = g.H() * g.W();
        const dim3 blk(T4_WARP_SQ, 1, 1);          ///< default blocks
        const dim3 grd((HW + blk.x - 1)/blk.x, g.C(), g.N());
        
        k_sgd<<<grd,blk>>>(
            g.data, dg.data, m.data, parm[0], parm[1], HW);
    };
    DU parm[3] = {
        lr / batch_size(),                        ///> eta / mini-batch size
        _iter ? b : (DU)DU0,                      ///> beta
        DU0
    };
    gradiant("sgd", update, parm, ABS(b) < DU_EPS ? OPTI_SGD : OPTI_SGDM);
    
    return *this;
}

__GPU__ Model&
Model::adam(DU lr, DU b1, DU b2) {
    auto update = [](DU *parm, Tensor &g, Tensor &dg, Tensor &m, Tensor &v) {
        const int HW = g.H() * g.W();
        const dim3 blk(T4_WARP_SQ, 1, 1);         ///< default blocks
        const dim3 grd((HW + blk.x - 1)/blk.x, g.C(), g.N());

        k_adam<<<grd,blk>>>(
            g.data, dg.data, m.data, v.data,
            parm[0], parm[1], parm[2], HW);
    };
    DU parm[3] = {
        lr * SQRT(1 - POW(b2, _iter+1)) / (1 - POW(b1, _iter+1)),
        _iter ? b1 : (DU)DU0,
        _iter ? b2 : (DU)DU0
    };
    gradiant("adam", update, parm, OPTI_ADAM);

    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
