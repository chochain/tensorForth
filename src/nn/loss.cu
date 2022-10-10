/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
__GPU__ Tensor&
Model::onehot() {
    auto show = [](DU *h, int n, int sz) {
/*                    
        printf("onehot[%d]=", n);
        for (int i = 0; i < sz; i++) {
            printf("%2.0f", h[i]);
        }
        printf("\n");
*/
    };
    Tensor &out = (*this)[-1];                         ///< model output
    int    N    = out.N(), hwc = out.HWC();            ///< sample size
    Tensor &hot = _t4(N, hwc).fill(DU0);               ///< one-hot vector
    if (!_dset) {
        ERROR("Model#loss dataset not set yet?\n");
        return hot;
    }
    for (int n = 0; n < N; n++) {                      /// * loop through batch
        DU *h = hot.slice(n);                          ///< take a sample
        U32 i = INT(_dset->label[n]);
        h[i < hwc ? i : 0] = DU1;
        show(h, n, hwc);
    }
    return hot;
}

__GPU__ DU
Model::loss(t4_loss op) {
    return loss(op, *_hot);                     /// * use default one-hot vector
}
__GPU__ DU
Model::loss(t4_loss op, Tensor &hot) {          ///< loss against one-hot
    Tensor &out = (*this)[-1];                  ///< model output
    if (!out.is_same_shape(hot)) {              /// * check dimensions
        ERROR("Model#loss hot dim != out dim\n");
        return;
    }
    Tensor &tmp = _mmu->copy(out);              ///< non-destructive
    DU err = _loss(op, tmp, hot);               /// * calculate loss
    _mmu->free(tmp);                            /// * free memory

    return err;
}
///
/// Stochastic Gradiant Decent
///
__GPU__ Model&
Model::sgd(DU lr, DU m, bool zero) {
    Tensor &n1 = (*this)[1];                    ///< reference model input layer
    ///
    /// cascade execution layer by layer forward
    ///
    auto trace = [](int i, Tensor &in) {
        printf("%2d> %s Σ/n=%6.2f [%d,%d,%d,%d]\tp=%-2d",
            i, d_nname(in.grad_fn), in.sum() / in.N() / in.C(),
            in.N(), in.H(), in.W(), in.C(), in.parm
            );
    };
    auto update = [lr, zero](Tensor &f, Tensor &df) {
        printf(" x[%d,%d,%d,%d]", f.N(), f.H(), f.W(), f.C());
        df *= lr;
        f  -= df;
        if (zero) df.map(O_FILL, DU0);
    };
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i];
        trace(i, in);
        if (in.grad[0] && in.grad[2]) {
            update(*in.grad[0], *in.grad[2]);
        }
        if (in.grad[1] && in.grad[3]) {
            update(*in.grad[1], *in.grad[3]);
        }
        printf("\n");
    }
    return *this;
}

__GPU__ Model&
Model::adam(DU lr, DU b0, DU b1, bool zero) {
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
