/**
 * @file
 * @brief tensorForth - Neural Network Model (i.e. Container in PyTorch)
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
    static __GPU__ void dconv2d(Tensor &A, Tensor &B)  {}
    static __GPU__ void drelu(Tensor &A, Tensor &B)    {}
    static __GPU__ void dmaxpool(Tensor &A, Tensor &B) {}
    static __GPU__ void dreshape(Tensor &A, Tensor &B) {}
    static __GPU__ void dlinear(Tensor &A, Tensor &B)  {}
    /// @}
    __GPU__ Tensor &operator[](int i) {
        return _mmu->du2ten(data[i]);
    }
    __GPU__ DU reset(MMU *mmu, Tensor &store) {
        _mmu   = mmu;
        _store = &store;
        data   = store.data;     // cached entries
        this->push(store);       // keep store as root
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
};
#endif // TEN4_SRC_MODEL_H
