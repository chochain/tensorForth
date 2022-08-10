/**
 * @file
 * @brief tensorForth - Neural Network Model (i.e. Container in PyTorch)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MODEL_H
#define TEN4_SRC_MODEL_H
#include "mmu.h"

#define NO_INIT  (!autograd || (nten->grad_fn != NONE))
class Model : public T4Base {
    MMU     *_mmu;       ///< tensor storage base
    Tensor  *_store;     ///< model storage - Sequential, TODO: DAG
    
public:
    bool    autograd = true;
    Tensor  *nten;       ///< cached tensor pointer
    ///
    /// @name Derivertive ops
    /// @{
    static __HOST__ const char* nname(int n);  /// network layer name
    static __GPU__ void dconv2d(Tensor &A, Tensor &B)  {}
    static __GPU__ void drelu(Tensor &A, Tensor &B)    {}
    static __GPU__ void dmaxpool(Tensor &A, Tensor &B) {}
    static __GPU__ void dflatten(Tensor &A, Tensor &B) {}
    static __GPU__ void dlinear(Tensor &A, Tensor &B)  {}
    /// @}
    __BOTH__ __INLINE__ Tensor &operator[](int i) {
        return _mmu->du2ten(data[i]);
    }
    __BOTH__ __INLINE__ void reset(MMU *mmu, Tensor &store) {
        _mmu   = mmu;
        _store = &store;
        size   = 0;
        dsize  = sizeof(DU);
        rank   = 0;
        ttype  = MODEL;
        data   = store.data;     // cached entries
        autograd = true;
        npush(store);            // keep store as root
    }
    __BOTH__ __INLINE__ DU npop() {
        DU n = data[--size];
        nten = &_mmu->du2ten(n);
        return n;
    }
    __BOTH__ __INLINE__ Model &npush(DU v) {
        nten = &_mmu->du2ten(v);
        data[size++] = v;
        return *this;
    }
    __BOTH__ __INLINE__ Model &npush(Tensor &t) {
        nten = &t;
        data[size++] = _mmu->ten2du(t);
        return *this;
    }
    __GPU__ __INLINE__ Tensor &tensor(U16 n, U16 h, U16 w, U16 c) {
        return _mmu->tensor(n, h, w, c);
    }
    __GPU__ __INLINE__ Tensor &vector(U16 n) {
        return _mmu->tensor(n);
    }
    /// @name Convolution and Linear initializer
    /// @{
    __GPU__ Model &iconv2d(DU bias, U16 c, U16 *opt);
    __GPU__ Model &ilinear(DU bias, U16 n);   ///< linearize (Dense) with n output
    __GPU__ Model &iflatten();      ///< flatten (input 
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ Model &irelu();         ///< Rectified Linear Unit
    __GPU__ Model &itanh();         ///< Tanh Unit
    __GPU__ Model &isigmoid();      ///< 1/(1+exp(-z))
    __GPU__ Model &isoftmax();      ///< probability vector exp(x)/sum(exp(x))
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __GPU__ Model &imaxpool(U16 n); ///< maximum pooling with nxn filter
    __GPU__ Model &iavgpool(U16 n); ///< average pooling with nxn filter
    __GPU__ Model &iminpool(U16 n); ///< minimum pooling with nxn filter
    __GPU__ Model &idropout(U16 p); ///< zero out p% of channel data (add noise between data points)
    /// @}
};
#endif // TEN4_SRC_MODEL_H
