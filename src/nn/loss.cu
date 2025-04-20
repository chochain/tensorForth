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
    return (*this)[-1];                             ///< return final output tensor
}
///
///> feed a tensor as the onehot vector
///
__GPU__ Tensor&
Model::onehot(Tensor &t) {
    Tensor &out = (*this)[-1];                      ///< model output
    U32    N    = out.N();                          ///< mini-batch size
    U32    E    = (U32)out.HWC();                   ///< channel sizes
    if (_hot) {
        ERROR("WARN: Model.onehot exists, replace with T%x\n", _mmu->OBJ2X(t));
        FREE(*_hot);
    }
    else if (t.N()!=N || (U32)t.HWC() != E) {
        ERROR("ERROR: onehot dimension is not [%d,%d,1,1]\n", N, E);
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
    U32    E    = (U32)out.HWC();                   ///< channel sizes
    Tensor &hot = T4(N, E).fill(DU0);               ///< one-hot vector
    auto show = [E](DU *h, U32 n, U32 m) {
        INFO("Model::onehot(ds) n=%d {", n);
        for (U32 e = 0; e < E; e++) {
            INFO("%2.0f%c", h[e], e==m ? '*' : ' ');
        }
        INFO(" }\n");
    };
    for (U32 n = 0; n < N; n++) {                   /// * loop through batch
        DU *h = hot.slice(n);                       ///< take a sample
        U32 m = dset.label[n];                      ///< label index
        h[m < E ? m : 0] = DU1;                     /// * mark hot by index
        if (*_trace > 1) show(h, n, m);             /// * might need U32 partition
    }
    return hot;
}

__GPU__ int
Model::hit(bool recalc) {
    if (!recalc) { return _hit; }                   /// * return current hit count
    
    Tensor &out = (*this)[-1];                      ///< model output
    U32    E    = (U32)out.HWC();                   ///< number of categories
    auto show = [E](DU *o, DU *h, U32 n, U32 m, U32 cnt) {
        for (U32 e = 0; e < E; e++) {
            INFO("%3.1f%c", o[e],
                 EQ(h[e],DU1) ? (e==m ? 'x' : '*') : (e==m ? '<' : ' '));
        }
        INFO(" n=%d cnt=%d\n", n, cnt);
    };
    auto argmax = [E](DU *o) {
        DU  m = o[0];
        U32 i = 0;
        for (U32 e = 1; e < E; e++) {               /// * CDP 
            if (o[e] > m) { m = o[e]; i = e; }
        }
        return i;
    };
    U32 cnt = 0;
    for (U32 n = 0; n < out.N(); n++) {             ///< loop through batch
        DU  *o = out.slice(n), *h = _hot->slice(n); ///< output vs onehot vectors
        U32  m = argmax(o);                         ///< index to max element
        cnt += D2I(h[m]);                           ///< lookup onehot vector
        if (*_trace > 1) show(o, h, n, m, cnt);
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
