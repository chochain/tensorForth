/**
 * @file
 * @brief tensorForth - Neural Network Model
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MODEL_H
#define TEN4_SRC_MODEL_H
#include "mmu.h"

class Model : public Managed {
    MMU     *_mmu;                ///< tensor storage base
    Tensor  *_store;              ///< model storage - Sequential, TODO: DAG

public:
    U16     idx = 0;              ///< model storage, Sequential stack, top
    DU      *data;                ///< cached data store
    Tensor  *nten;                ///< cached tensor pointer
    ///
    /// @name Derivertive ops
    /// @{
    static __BOTH__ void dconv2d(Tensor &A, Tensor &B)  {}
    static __BOTH__ void drelu(Tensor &A, Tensor &B)    {}
    static __BOTH__ void dmaxpool(Tensor &A, Tensor &B) {}
    static __BOTH__ void dreshape(Tensor &A, Tensor &B) {}
    static __BOTH__ void dlinear(Tensor &A, Tensor &B)  {}
    static __BOTH__ const char *fname(GradFn f) {
        if (f == dconv2d) return "conv2d ";
        else              return "input  ";
    }
    /// @}
    __GPU__ DU reset(MMU *mmu, Tensor &store) {
        _mmu   = mmu;
        _store = &store;
        data  = (DU*)store.data;
        push(store);
    }
    __GPU__ DU pop() {
        DU n = data[--idx];
        nten = &_mmu->du2ten(n);
        return n;
    }
    __GPU__ DU push(DU v) {
        nten = &_mmu->du2ten(v);
        return data[idx++] = v;
    }
    __GPU__ DU push(Tensor &t) {
        nten = &t;
        return data[idx++] = _mmu->ten2du(t);
    }
    __GPU__ void init_conv2d(DU bias, IU c, U16 *opt) {
        Tensor &in = *nten;
        if (in.grad_fn) return;
        
        in.grad_fn = &dconv2d;                       ///> derivative function
        
        U16 m = opt[0], n = opt[1];                  ///> filter sizing
        U16 p = opt[2] ? opt[2] : floor((m-1)/2);    ///> padding
        U16 s = opt[3], d = opt[4];                  ///> stride, dilation
        
        Tensor *w  = in.grad[0] = &_mmu->tensor(1, m, n, c);                 ///> w
        Tensor *b  = in.grad[1] = &_mmu->tensor(1, 1, 1, c).map(FILL, bias); ///> b
        Tensor *dw = in.grad[2] = &_mmu->tensor(1, m, n, c).map(FILL, DU0);  ///> dw
        Tensor *db = in.grad[3] = &_mmu->tensor(1, 1, 1, c).map(FILL, DU0);  ///> db
    
        _mmu->random(*w, NORMAL);                      /// * randomize w
        Tensor &out = _mmu->tensor(1, m, n, c);
        push(out);
    }
};
#endif // TEN4_SRC_MODEL_H
