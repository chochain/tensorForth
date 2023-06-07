/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
__GPU__ DU
Model::_loss(t4_loss op, Tensor &out, Tensor &hot) {
    const int N = out.N();
    DU  err = DU0;                   ///> result loss value
    switch (op) {
    case LOSS_MSE:                   /// * mean squared error, input from linear
        out -= hot;
        err = 0.5 * NORM(out.numel, out.data) / N;
        break;
    case LOSS_BCE:                   /// * binary cross_entropy, input from sigmoid
        out -= hot;
        err  = -out.sum() / N;
        break;
    case LOSS_CE:                    /// * cross_entropy, input from softmax
        out.map(O_LOG);
        /* no break */
    case LOSS_NLL:                   /// * negative log likelihood, input from log-softmax
        out *= hot;                  /// * hot_i * log(out_i)
        err = -out.sum() / N;        /// * negative average per sample
        break;
    default: ERROR("Model#loss op=%d not supported!\n", op);
    }
    // debug(out);
    SCALAR(err);
    return err;
}

__GPU__ Tensor&
Model::onehot() {
    if (_hot) return *_hot;
    
    ERROR("Model.onehot not initialized, run forward first!\n");
    return (*this)[-1];
}

__GPU__ Tensor&
Model::onehot(Dataset &dset) {
    auto show = [](DU *h, int n, int sz) {
        printf("Model::onehot[%d]={", n);
        for (int i = 0; i < sz; i++) {
            printf("%2.0f", h[i]);
        }
        printf("}\n");
    };
    Tensor &out = (*this)[-1];                      ///< model output
    int    N    = out.N(), hwc = out.HWC();         ///< sample size
    Tensor &hot = _t4(N, hwc).fill(DU0);            ///< one-hot vector
    for (int n = 0; n < N; n++) {                   /// * loop through batch
        DU *h = hot.slice(n);                       ///< take a sample
        U32 i = INT(dset.label[n]);
        h[i < hwc ? i : 0] = DU1;
        if (_trace > 1) show(h, n, hwc);
    }
    return hot;
}

__GPU__ int
Model::hit(bool recalc) {
    if (!recalc) { return _hit; }                   /// * return current hit count
    
    auto argmax = [](DU *h, int sz) {
        DU  mx = *h;
        int m  = 0;
        for (int i = 1; i < sz; i++) {              /// * CDP 
            if (h[i] > mx) { mx = h[i]; m = i; }
        }
        return m;
    };
    Tensor &out = (*this)[-1];                      ///< model output
    int cnt = 0;
    for (int n = 0; n < out.N(); n++) {             ///< loop through batch
        int  m = argmax(out.slice(n), out.HWC());
        cnt += INT(_hot->slice(n)[m]);              /// * compare to onehot vector
    }
    TRACE1("Model::hit=%d\n", cnt);
    return cnt;
}

__GPU__ DU
Model::loss(t4_loss op) {
    return loss(op, *_hot);                         /// * use default one-hot vector
}

__GPU__ DU
Model::loss(t4_loss op, Tensor &hot) {              ///< loss against one-hot
    Tensor &out = (*this)[-1];                      ///< model output
    if (!out.is_same_shape(hot)) {                  /// * check dimensions
        ERROR("Model#loss hot dim != out dim\n");
        return DU0;
    }
    Tensor &tmp = _mmu->copy(out);                  ///< non-destructive
    DU err = _loss(op, tmp, hot);                   /// * calculate loss
    _mmu->free(tmp);                                /// * free memory

    return err;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
