/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)
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
    auto show = [](DU *h, U32 n, U64 sz) {
        INFO("Model::onehot[%d]={", n);
        for (U64 i = 0; i < sz; i++) {
            INFO("%2.0f", h[i]);
        }
        INFO(" }\n");
    };
    Tensor &out = (*this)[-1];                      ///< model output
    U32    N    = out.N();
    U64    HWC  = out.HWC();                        ///< sample size
    Tensor &hot = T4(N, HWC).fill(DU0);             ///< one-hot vector
    for (U32 n = 0; n < N; n++) {                   /// * loop through batch
        DU *h = hot.slice(n);                       ///< take a sample
        U32 i = dset.label[n];                      ///< label index
        h[(U64)i < HWC ? i : 0] = DU1;              /// * mark hot by index
        if (*_trace > 1) show(h, n, HWC);           /// * might need U32 partition
    }
    return hot;
}

__GPU__ int
Model::hit(bool recalc) {
    if (!recalc) { return _hit; }                   /// * return current hit count
    
    auto argmax = [](DU *h, U64 sz) {
        DU  mx = *h;
        U32 m  = 0;
        for (U64 i = 1; i < sz; i++) {              /// * CDP 
            if (h[i] > mx) { mx = h[i]; m = i; }
        }
        return m;
    };
    Tensor &out = (*this)[-1];                      ///< model output
    U32 cnt = 0;
    for (U32 n = 0; n < out.N(); n++) {             ///< loop through batch
        U32  m = argmax(out.slice(n), out.HWC());
        cnt += INT(_hot->slice(n)[m]);              /// * compare to onehot vector
    }
    NN_DB("Model::hit=%d\n", cnt);
    return cnt;
}

__GPU__ DU
Model::loss(t4_loss op) {
    return loss(op, *_hot);                         /// * use default one-hot vector
}

__GPU__ DU
Model::loss(t4_loss op, Tensor &tgt) {              ///< loss against target vector
    static const char *_op[] = { "MSE", "BCE", "CE", "NLL" };
    Tensor &out = (*this)[-1];                      ///< model output
    if (out.numel != tgt.numel) {                   /// * check dimensions
        ERROR("Model::loss model output shape[%d,%d,%d,%d] != tgt[%d,%d,%d,%d]\n",
            out.N(), out.H(), out.W(), out.C(),
            tgt.N(), tgt.H(), tgt.W(), tgt.C());
        return DU0;
    }
    Tensor &tmp = COPY(out);                        ///< non-destructive
    DU sum = tmp.loss(op, tgt);                     /// * calculate loss per op
    FREE(tmp);                                      /// * free memory
    
    NN_DB("Model#loss: %s=%6.3f\n", _op[op], sum);
    
    return sum;
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
