/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
#include "model.h"
#include "dataset.h"

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
    int    N    = out.N(), HWC = out.HWC();         ///< sample size
    Tensor &hot = _t4(N, HWC).fill(DU0);            ///< one-hot vector
    for (int n = 0; n < N; n++) {                   /// * loop through batch
        DU *h = hot.slice(n);                       ///< take a sample
        U16 i = dset.label[n];                      ///< label index
        h[i < HWC ? i : 0] = DU1;                   /// * mark hot by index
        if (_mmu->trace() > 1) show(h, n, HWC);
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
Model::loss(t4_loss op, Tensor &tgt) {              ///< loss against target vector
    static const char *opn[] = { "MSE", "BCE", "CE", "NLL" };
    Tensor &out = (*this)[-1];                      ///< model output
    if (!out.is_same_shape(tgt)) {                  /// * check dimensions
        ERROR("Model#loss: hot dim != out dim\n");
        return DU0;
    }
    Tensor &tmp = _mmu->copy(out);                 ///< non-destructive
    DU sum = tmp.loss(op, tgt);                    /// * calculate loss per op
    _mmu->free(tmp);                               /// * free memory
    
    TRACE1("Model#loss: %s=%6.3f\n", opn[op], sum);
    
    return sum;
}
#endif  // (T4_ENABLE_OBJ && T4_ENABLE_NN)
//==========================================================================
