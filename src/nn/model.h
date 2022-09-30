/**
 * @file
 * @brief tensorForth - Neural Network Model (i.e. Container in PyTorch)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MODEL_H
#define TEN4_SRC_MODEL_H
#include "mmu.h"         // in ../mmu

typedef enum {
    LOSS_MSE = 0,        ///< mean square error
    LOSS_NLL,            ///< negative likelihood
    LOSS_CE              ///< cross entropy
} t4_loss;

class Model : public T4Base {
    MMU     *_mmu;       ///< tensor storage base
    Tensor  *_store;     ///< model storage - Sequential, TODO: DAG
    
public:
    bool    autograd = true;
    ///
    /// @name Derivertive ops
    /// @{
    static __HOST__ const char* nname(int n);    /// network layer name
    static __GPU__  const char* d_nname(int n);
    /// @}
    __BOTH__ __INLINE__ Tensor &operator[](int i) {
        return (Tensor&)_mmu->du2obj(data[(i < 0) ? numel + i : i]);
    }
    __BOTH__ __INLINE__ int  slots() { return _store->numel; }
    __GPU__  __INLINE__ void reset(MMU *mmu, Tensor &store) {
        _mmu   = mmu;
        _store = &store;
        numel  = 0;
        dsize  = DSIZE;
        rank   = 0;
        ttype  = T4_MODEL;
        data   = store.data;     // cached entries
        autograd = true;
        npush(store);            // keep store as root
    }
    __GPU__ __INLINE__ Model &npush(DU v) {
        data[numel++] = v;
        U32 tsz = _store->numel;
        if (tsz <= numel) {
            _mmu->resize(*_store, tsz + T4_NET_SZ);
            data = _store->data; // reset storage cached pointer
        }
        return *this;
    }
    __GPU__ __INLINE__ Model  &npush(Tensor &t) { return npush(_mmu->obj2du(t)); }
    __GPU__ __INLINE__ DU     npop() { return data[--numel]; }
    __GPU__ __INLINE__ Tensor &output() { return (*this)[numel-1]; }
    __GPU__ __INLINE__ Tensor &vector(U16 sz) {
        return _mmu->tensor(sz);
    }
    __GPU__ __INLINE__ Tensor &tensor(U16 n, U16 h, U16 w, U16 c) {
        return _mmu->tensor(n, h, w, c);
    }
    __GPU__ __INLINE__ Tensor &tensor(U16 c1, U16 n, U16 h, U16 w, U16 c) {
        return _mmu->tensor(c1, n, h, w, c);
    }
    __GPU__ Model  &add(t4_layer fn, U16 n=0, DU bias=DU0, U16 *opt=0);
    __GPU__ Model  &forward(Tensor &input);
    __GPU__ Model  &backprop(Tensor &tgt);
    __GPU__ DU     loss(t4_loss op, Tensor &tgt);
    ///
    /// debug dump
    ///
    __GPU__ void   view(DU *v, int H, int W, int C);
    __GPU__ void   dump(DU *v, int H, int W, int C);
    __GPU__ void   dump_dbdf(DU *df, DU *db, int C0, int C1, int fsz);

private:
    /// @name single step forward and backprop
    /// @{
    __GPU__ void _fstep(Tensor &in, Tensor &out);
    __GPU__ void _bstep(Tensor &in, Tensor &out);
    /// @}
    /// @name Convolution and Linear initializer
    /// @{
    __GPU__ void _iconv(Tensor &in, U16 c, DU bias, U16 *opt);
    __GPU__ void _ilinear(Tensor &in, U16 n, DU bias);   ///< linearize (Dense) with n output
    __GPU__ void _iflatten(Tensor &in);        ///< flatten (input 
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ void _icopy(Tensor &in);           ///< Relu, Tanh, Sigmoid
    __GPU__ void _isoftmax(Tensor &in);        ///< probability vector exp(x)/sum(exp(x))
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __GPU__ void _ipool(Tensor &in, U16 n);    ///< maximum pooling with nxn filter
    __GPU__ void _idropout(Tensor &in, U16 p); ///< zero out p% of channel data (add noise between data points)
    /// @}
};
#endif // TEN4_SRC_MODEL_H
