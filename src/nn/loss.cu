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
    
    ERROR("ERROR: Model.onehot not provided by dataset, use nn.onehot= to setup!\n");
    return (*this)[-1];
}
///
///> feed a tensor as the onehot vector
///
__GPU__ Tensor&
Model::onehot(Tensor &t) {
    Tensor &out = (*this)[-1];                      ///< model output
    U32    N    = out.N();                          ///< mini-batch size
    U32    C    = (U32)out.HWC();                   ///< channel sizes
    if (_hot) {
        ERROR("WARN: Model.onehot exists, replace with T%x\n", _mmu->OBJ2X(t));
        FREE(*_hot);
    }
    else if (t.N()!=N || (U32)t.HWC() != C) {
        ERROR("ERROR: onehot dimension is not [%d,%d,1,1]\n", N, C);
        return t;
    }
    _hot = &t;                                      ///< assign onehot vector
    _hit = hit(true);                               ///< calculate hit counts
    
    return *_hot;
}
///
///> capture onehot vector from dataset labels
///
__GPU__ Tensor&
Model::onehot(Dataset &dset) {
    Tensor &out = (*this)[-1];                      ///< model output
    U32    N    = out.N();                          ///< mini-batch size
    U32    C    = (U32)out.HWC();                   ///< channel sizes
    Tensor &hot = T4(N, C).fill(DU0);               ///< one-hot vector
    auto show = [C](DU *h, U32 n) {
        INFO("Model::onehot[%d]={", n);
        for (U32 c = 0; c < C; c++) {
            INFO("%2.0f", h[c]);
        }
        INFO(" }\n");
    };
    for (U32 n = 0; n < N; n++) {                   /// * loop through batch
        DU *h = hot.slice(n);                       ///< take a sample
        U32 c = dset.label[n];                      ///< label index
        h[c < C ? c : 0] = DU1;                     /// * mark hot by index
        if (1 || *_trace > 1) show(h, n);             /// * might need U32 partition
    }
    return hot;
}

__GPU__ int
Model::hit(bool recalc) {
    if (!recalc) { return _hit; }                   /// * return current hit count
    
    Tensor &out = (*this)[-1];                      ///< model output
    U32    C    = (U32)out.HWC();                   ///< number of channels
    auto argmax = [C](DU *h) {
        DU  mx = *h;
        U32 m  = 0;
        for (U32 c = 1; c < C; c++) {               /// * CDP 
            if (h[c] > mx) { mx = h[c]; m = c; }
        }
        return m;
    };
    U32 cnt = 0;
    for (U32 n = 0; n < out.N(); n++) {             ///< loop through batch
        U32 m = argmax(out.slice(n));               ///< index to max element
        U32 v = UINT(_hot->slice(n)[m]);            ///< lookup onehot vector
        cnt += v;                                   /// * acculate hit count
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
    DU z = tmp.loss(op, tgt);                       /// * calculate loss per op
    FREE(tmp);                                      /// * free memory
    
    NN_DB("Model#loss: %s=%6.3f\n", _op[op], z);
    
    return z;
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
