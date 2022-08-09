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
    static __GPU__ void dflatten(Tensor &A, Tensor &B) {}
    static __GPU__ void dlinear(Tensor &A, Tensor &B)  {}
    /// @}
    __GPU__ __INLINE__ bool   not_set() { return nten->grad_fn == NONE; } // not set
    __GPU__ __INLINE__ Tensor &operator[](int i) {
        return _mmu->du2ten(data[i]);
    }
    __GPU__ __INLINE__ Tensor &tensor(U16 n, U16 h, U16 w, U16 c) {
        return _mmu->tensor(n, h, w, c);
    }
    __GPU__ __INLINE__ Tensor &vector(U16 n) {
        return _mmu->tensor(n);
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
    /// @name Convolution and Linear initializer
    /// @{
    __GPU__ void iconv2d(DU bias, U16 c, U16 *opt);
    __GPU__ void ilinear(DU bias, U16 n);   ///< linearize (Dense) with n output
    __GPU__ void iflatten();
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ void irelu();          ///< Rectified Linear Unit
    __GPU__ void itanh();          ///< Tanh Unit
    __GPU__ void isigmoid();       ///< 1/(1+exp(-z))
    __GPU__ void isoftmax();       ///< probability vector exp(x)/sum(exp(x))
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __GPU__ void imaxpool(U16 n);  ///< maximum pooling with nxn filter
    __GPU__ void iavgpool(U16 n);  ///< average pooling with nxn filter
    __GPU__ void iminpool(U16 n);  ///< minimum pooling with nxn filter
    __GPU__ void idropout(U16 p);  ///< zero out p% of channel data (add noise between data points)
    /// @}
};
#endif // TEN4_SRC_MODEL_H
