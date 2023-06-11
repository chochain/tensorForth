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
    Tensor &tmp = _mmu->copy(out);   ///< non-destructive
    int N   = tmp.N();               ///< mini-batch sample count
    DU  sum = DU0;                   ///> result loss value
    switch (op) {
    case LOSS_MSE:                   /// * mean squared error, input from linear
        tmp -= hot;
        sum = 0.5 * NORM(tmp.numel, tmp.data);
        break;
    case LOSS_BCE:                   /// * binary cross_entropy, input from sigmoid
        for (int n=0; n < N; n++) {
            int k = n * tmp.HWC();
            DU  p = hot.data[k], q = tmp.data[k];
            sum -= p * LOG(q) + (DU1 - p) * LOG(DU1 - q);
        }
        break;
    case LOSS_CE:                    /// * cross_entropy, input from softmax
        tmp.map(O_LOG);
        /* no break */
    case LOSS_NLL:                   /// * negative log likelihood, input from log-softmax
        tmp *= hot;                  /// * hot_i * log(out_i)
        sum = -tmp.sum();            /// * negative sum
        break;
    default: ERROR("Model#loss op=%d not supported!\n", op);
    }
    // debug(tmp);
    _mmu->free(tmp);                 /// * free memory

    sum /= N;                        /// average per mini-batch sample
    return SCALAR(sum);              /// make sum a scalar value (not object)
}

__GPU__ Tensor&
Model::onehot() {
    if (_hot) return *_hot;
    
    ERROR("ERROR: Model.onehot not provided by dataset, input onehot tensor!\n");
    return (*this)[-1];
}
///
///> capture onehot vector from dataset labels
///
__GPU__ Tensor&
Model::onehot(Dataset &dset) {
    auto show = [](DU *h, int n, int sz) {
        printf("Model::onehot[%d]={", n);
        for (int i = 0; i < sz; i++) {
            printf("%2.0f", h[i]);
        }
        printf(" }\n");
    };
    Tensor &out = (*this)[-1];                      ///< model output
    int    N    = out.N(), hwc = out.HWC();         ///< sample size
    Tensor &hot = _t4(N, hwc).fill(DU0);            ///< one-hot vector
    for (int n = 0; n < N; n++) {                   /// * loop through batch
        DU *h = hot.slice(n);                       ///< take a sample
        U32 i = INT(dset.label[n]);                 ///< label index
        h[i < hwc ? i : 0] = DU1;                   /// * mark hot by index
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
    return _loss(op, out, hot);                     /// * calculate loss
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
