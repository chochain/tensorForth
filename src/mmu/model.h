/**
 * @file
 * @brief Model class - NN model (i.e. Container in PyTorch)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MODEL_H
#define TEN4_SRC_MODEL_H
#include "mmu.h"         // in ../mmu

typedef enum {
    UP_NEAREST = 0,
    UP_LINEAR,
    UP_BILINEAR,
    UP_CUBIC
} t4_upsample;

typedef enum {
    OPTI_SGD = 0,            ///< Stochastic Gradient Descent
    OPTI_SGDM,               ///< SGD with momemtum
    OPTI_ADAM                ///< Adam gradiant
} t4_optimizer;
///
///< gradiant function pointer
///
typedef void (*GdFunc)(
    DU *parm, Tensor &w, Tensor &dw, Tensor &m, Tensor &v);
///
///< Neural Network Model class
///
class Model : public T4Base {
    MMU          *_mmu;              ///< tensor storage base
    Tensor       *_store;            ///< model storage - Sequential, TODO: DAG
    Tensor       *_hot  = NULL;      ///< cached dataset one-hot vector
    int          _trace = 0;         ///< cached debug/tracing level
    int          _iter  = 0;         ///< iteration counter
    int          _hit   = 0;         ///< hit counter
    t4_optimizer _opti  = OPTI_SGD;  ///< optimizer selection
    DU           _gparm[3];          ///< gradiant parameters
    
public:
    bool autograd = true;
    ///
    /// @name Derivertive ops
    /// @{
    static __HOST__ const char* nname(int n);    ///< network layer name on host
    static __GPU__  const char* d_nname(int n);  ///< network layer name on device
    /// @}
    /// @name layer access methods
    /// @{
    __BOTH__ __INLINE__ Tensor &operator[](int i) {
        /// * model.data[0] = store
        /// so 1st layer starts from model.data[1]
        return (Tensor&)_mmu->du2obj(data[(i < 0) ? numel + i : i]);
    }
    __BOTH__ __INLINE__ int  slots() { return _store->numel; }
    __GPU__  __INLINE__ void reset(MMU *mmu, Tensor &store, int trace) {
        init(0, T4_MODEL, 0);                   /// * T4Base attributes
        _trace = trace;
        _mmu   = mmu;
        _store = &store;
        data   = store.data;                    /// * cached entries
        autograd = true;
        npush(store);                           /// * model.data[0] = store
    }
    __GPU__ __INLINE__ Model &npush(DU v) {
        data[numel++] = v;
        U32 tsz = _store->numel;                ///< current allocated for layers
        if (tsz <= numel) {                     /// * resize if too many layers
            _mmu->resize(*_store, tsz + T4_NET_SZ);
            data = _store->data;                /// * reset storage cached pointer
        }
        return *this;
    }
    __GPU__ __INLINE__ Model  &npush(Tensor &t) { return npush(_mmu->obj2du(t)); }
    __GPU__ __INLINE__ DU     npop() { return data[--numel]; }
    __GPU__ __INLINE__ int    batch_size() { return (*this)[1].N(); }
    /// @}
    /// @name main NN methods
    /// @{
    __GPU__ Model  &add(t4_layer fn, U16 n=0, DU bias=DU0, U16 *opt=0);
    __GPU__ Model  &forward(Tensor &input);             ///< network feed forward
    __GPU__ Model  &broadcast(Tensor &tgt);
    __GPU__ Model  &backprop();                         ///< back propegation with default onehot vector (built during forward pass from dataset labels)
    __GPU__ Model  &backprop(Tensor &tgt);              ///< back propegation with given target vector
    /// @}
    /// @name loss functions
    /// @{
    __GPU__ Tensor &onehot();                           ///< get default onehot vector
    __GPU__ Tensor &onehot(Dataset &dset);              ///< create one-hot vector from dataset labels (called in forward pass)
    __GPU__ int    hit(bool recalc=false);              ///< calculate hit count
    __GPU__ DU     loss(t4_loss op);                    ///< calc loss with cached one-hot vector
    __GPU__ DU     loss(t4_loss op, Tensor &tgt);       ///< calc loss from tgt vector
    /// @}
    /// @name gradiant decent functions
    /// @{
    __GPU__ Model  &grad_alloc(Tensor &in, bool do_w, bool do_b);
    __GPU__ Model  &grad_zero() { _iter = 0; }          ///< manual zero momentum tensors
    __GPU__ Model  &gradiant(const char *nm, GdFunc fn);///< gradiant descent functor
    __GPU__ Model  &sgd(DU lr, DU b);                   ///< stochastic gradiant descent
    __GPU__ Model  &adam(DU lr, DU b1, DU b2);          ///< Adam gradiant descent
    /// @}
    /// @name debug functions
    /// @{
    __GPU__ void   debug(Tensor &t, DU scale=10.0f);
    /// @}

private:
    /// @name internal tensor constructors
    /// @{
    __GPU__ __INLINE__ Tensor &_vec(U16 sz)                            { return _mmu->tensor(sz); }
    __GPU__ __INLINE__ Tensor &_t4(U16 n, U16 h)                       { return _mmu->tensor(n, h, 1, 1); }
    __GPU__ __INLINE__ Tensor &_t4(U16 n, U16 h, U16 w, U16 c)         { return _mmu->tensor(n, h, w, c); }
    /// @}
    /// @name Convolution and Linear initializer
    /// @{
    __GPU__ void   _iconv(Tensor &in, U16 c, DU bias, U16 *opt);
    __GPU__ void   _ilinear(Tensor &in, U16 n, DU bias);   ///< linearize (Dense) with n output
    __GPU__ void   _iflatten(Tensor &in);           ///< flatten (input 
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ void   _icopy(Tensor &in);              ///< for relu, tanh, sigmoid
    __GPU__ void   _iactivate(Tensor &in, DU alpha);///< zero out p% of channel data (add noise between data points)
    __GPU__ void   _isoftmax(Tensor &in);           ///< for softmax, logsoftmax
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __GPU__ void   _ipool(Tensor &in, U16 n);       ///< pooling with nxn filter
    __GPU__ void   _idropout(Tensor &in, DU pct);   ///< zero out p% of channel data (add noise between data points)
    __GPU__ void   _iup(Tensor &in, U16 n, DU m);   ///< upsample with nxn filter
    __GPU__ void   _ibatchnorm(Tensor &in, DU m);   ///< batch norm with momentum=m
    /// @}
    /// @name forward ops
    /// @{
    __GPU__ void   _fstep(Tensor &in, Tensor &out);
    __GPU__ int    _fconv(Tensor &in, Tensor &out);
    __GPU__ int    _flinear(Tensor &in, Tensor &out);
    __GPU__ int    _ffilter(Tensor &in, Tensor &m, Tensor &out);
    __GPU__ int    _factivate(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _fpool(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _fsoftmax(Tensor &in, Tensor &out);
    __GPU__ int    _flogsoftmax(Tensor &in, Tensor &out);
    __GPU__ int    _fupsample(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _fbatchnorm(Tensor &in, Tensor &out);
    /// @}
    /// @name backward ops
    /// @{
    __GPU__ void   _bstep(Tensor &in, Tensor &out);
    __GPU__ int    _bconv(Tensor &in, Tensor &out);
    __GPU__ int    _blinear(Tensor &in, Tensor &out);
    __GPU__ int    _bfilter(Tensor &in, Tensor &msk, Tensor &out);
    __GPU__ int    _bpool(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _bupsample(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _bbatchnorm(Tensor &in, Tensor &out);
    /// @}
    /// @name debug functions
    /// @{
    __GPU__ void   _view(DU *v, int H, int W, int C, DU scale=10.0f);
    __GPU__ void   _dump(DU *v, int H, int W, int C);
    __GPU__ void   _dump_dbdf(Tensor &db, Tensor &df);
    __GPU__ void   _dump_db(Tensor &db);
    __GPU__ void   _dump_dw(Tensor &dw, bool full=false);
    /// @}
};
#endif // TEN4_SRC_MODEL_H
