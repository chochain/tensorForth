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
/// Note: does not get affected by batch size
///       because filters are fixed size
///
__GPU__ Model&
Model::sgd(DU lr, DU m, bool zero) {
    Tensor &n1 = (*this)[1];                   ///< reference model input layer
    DU     t0  = _mmu->ms();                   ///< performance measurement
    ///
    /// cascade execution layer by layer forward
    ///
    const int N = n1.N();                      ///< batch size
    auto update = [this, N, lr, zero](const char nm, Tensor &f, Tensor &df) {
        TRACE1(" %c[%d,%d,%d,%d]", nm, f.N(), f.H(), f.W(), f.C());
        df *= lr / N;                          /// * learn rate / batch size
        f  -= df;                              /// * w -= eta * df, TODO: momentum
        if (zero) df.map(O_FILL, DU0);         /// * zap df, ready for next batch
//        debug(f);
    };
    TRACE1("\nModel#sgd batch_sz=%d, lr=%6.3f", N, lr);
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i];

        TRACE1("\n%2d> %s ", i, d_nname(in.grad_fn));
        if (in.grad[0] && in.grad[2]) {
            update('f', *in.grad[0], *in.grad[2]);
        }
        if (in.grad[1] && in.grad[3]) {
            update('b', *in.grad[1], *in.grad[3]);
        }
    }
    TRACE1("\nModel#sgd %5.2f ms\n", _mmu->ms() - t0);
    return *this;
}

__GPU__ Model&
Model::adam(DU lr, DU b0, DU b1, bool zero) {
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
