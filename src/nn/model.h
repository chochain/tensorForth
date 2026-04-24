/**
 * @file
 * @brief Model class - NN model (i.e. Container in PyTorch)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#pragma once
#include "ten4_config.h"

#if (!defined(__MMU_MODEL_H) && T4_DO_OBJ && T4_DO_NN)
#define __MMU_MODEL_H
#include "sys.h"             /// * ms, rand
#include "mu/mmu.h"

namespace t4::mu { class Tensor; class Dataset; }
namespace t4::nn {

typedef enum {
    UP_NEAREST = 0,
    UP_LINEAR,
    UP_BILINEAR,
    UP_CUBIC
} t4_upsample;

typedef enum {
    OPTI_SGD = 0,            ///< Stochastic Gradient Descent
    OPTI_SGDM,               ///< SGD with momemtum
    OPTI_ADAM                ///< Adam gradient
} t4_optimizer;
///
///< gradient function pointer
///
typedef void (*GdFunc)(
    DU *parm, mu::Tensor &w, mu::Tensor &dw, mu::Tensor &m, mu::Tensor &v);
///
///< Neural Network Model class
///
#define NLOG(...)             { if (*_trace) INFO(__VA_ARGS__); }

class Model : public T4Base {
    using MMU     = mu::MMU;     ///< alias
    using Tensor  = mu::Tensor;
    using Dataset = mu::Dataset;

    MMU    *_mmu;                ///< memory controller
    Tensor *_store;              ///< model storage - Sequential, TODO: DAG
    Tensor *_hot    = NULL;      ///< cached dataset one-hot vector
    Tensor *_loss   = NULL;      ///< cached dataset loss vector
    int    _hit     = 0;         ///< hit counter
    int    _iter    = 0;         ///< iteration counter (for Adam)
    int    *_trace;              ///< trace level
    int    _err     = 0;
    
public:
    DU     max_norm = DU0;       ///< gradient clipping
    int    epoch    = 0;         ///< TODO: for learning rate decay
    ///
    /// @name Derivertive ops
    /// @{
    static __HOST__ const char* nname(int n);    ///< network layer name on host
    /// @}
    /// @name constructor (indirect)
    /// @{
    __HOST__  void   init(MMU *mmu, Tensor &store, int &trace);
    __HOST__  void   tick() { epoch++; _iter=0; } ///< advance epoch counter
    /// @}
    /// @name layer access methods
    /// @{
    __HOST__ Tensor &operator[](S64 i);          ///< 64-bit indexing (negative possible)
    __HOST__ int    slots();
    
    __HOST__  Model  &npush(DU v);
    __HOST__  Model  &npush(Tensor &t);
    __HOST__  DU     npop();
    __HOST__  int    batch_size();
    /// @}
    /// @name Tensor constructors and randomizer
    /// @{
    __HOST__ Tensor &COPY(Tensor &t);                    ///< hardcopy a tensor (proxy to mmu)
    __HOST__ void   FREE(Tensor &t);
    __HOST__ Tensor &VEC(U64 sz);                        ///< proxy to MMU::tensor
    __HOST__ Tensor &T4(U32 n, U32 h);
    __HOST__ Tensor &T4(U32 n, U32 h, U32 w, U32 c);
    __HOST__ void   RAND(Tensor &t, DU scale);           ///< proxy to System::rand
    /// @}
    /// @name main NN methods
    /// @{
    __HOST__ Model  &add(t4_layer fn, U32 n=0, DU alpha=DU0, U16 *opt=NULL);
    __HOST__ Model  &forward(Tensor &input);             ///< network feed forward
    __HOST__ Model  &broadcast(Tensor &tgt);
    __HOST__ Model  &backprop();                         ///< back propegation with default onehot vector (built during forward pass from dataset labels)
    __HOST__ Model  &backprop(Tensor &tgt);              ///< back propegation with given target vector
    /// @}
    /// @name loss functions
    /// @{
    __HOST__ Tensor &onehot();                           ///< get default onehot vector
    __HOST__ Tensor &onehot(Tensor &t);                  ///< feed tensor as the one-hot vector
    __HOST__ Tensor &onehot(Dataset &dset);              ///< create one-hot vector from dataset labels (called in forward pass)
    __HOST__ int    hit(bool recalc=true);               ///< calculate hit count
    __HOST__ DU     loss(t4_loss op);                    ///< calc loss with cached one-hot vector
    __HOST__ DU     loss(t4_loss op, Tensor &tgt);       ///< calc loss from tgt vector
    /// @}
    /// @name gradient descent functions
    /// @{
    __HOST__ Model  &grad_zero() { _iter = _hit = 0; return *this; }
    __HOST__ Model  &grad_alloc(t4_optimizer op);        ///< allocate gradient vectors
    __HOST__ Model  &gradient(const char *nm,            ///< gradient descent functor
                             t4_optimizer op,
                             GdFunc fn,                 
                             DU *parm);
    __HOST__ Model  &sgd(DU lr, DU b=0.9);               ///< stochastic gradient descent
    __HOST__ Model  &adam(DU lr, DU b1=0.9, DU b2=0.999);///< Adam gradient descent
    /// @}
    
private:
    /// @name Convolution and Linear initializer
    /// @{
    __HOST__ void   _iconv(Tensor &in, U32 c, DU bias, U16 *opt);    ///< 2D convolution
    __HOST__ void   _ilinear(Tensor &in, U32 n, DU bias);            ///< linearize (Dense) with n output
    __HOST__ void   _iflatten(Tensor &in);                           ///< flatten
    /// @}
    /// @name Activation ops
    /// @{
    __HOST__ void   _isoftmax(Tensor &in);                           ///< for softmax, logsoftmax
    __HOST__ void   _iactivate(Tensor &in, DU alpha);                ///< relu, tanh, sigmoid, zero out p% of channel data (add noise between data points)
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __HOST__ void   _ipool(Tensor &in, U16 f);       ///< pooling with nxn filter
    __HOST__ void   _ibatchnorm(Tensor &in, DU m);   ///< batch norm with momentum=m
    __HOST__ void   _iup(Tensor &in, U16 f, DU m);   ///< upsample with nxn filter
    /// @}
    /// @name forward ops
    /// @{
    __HOST__ void   _fstep(Tensor &in, Tensor &out);
    __HOST__ int    _fconv(Tensor &in, Tensor &out);
    __HOST__ int    _flinear(Tensor &in, Tensor &out);
    __HOST__ int    _factivate(Tensor &in, Tensor &out, t4_layer fn);
    __HOST__ int    _fpool(Tensor &in, Tensor &out, t4_layer fn);
    __HOST__ int    _fsoftmax(Tensor &in, Tensor &out);
    __HOST__ int    _flogsoftmax(Tensor &in, Tensor &out);
    __HOST__ int    _fbatchnorm(Tensor &in, Tensor &out);
    __HOST__ int    _fupsample(Tensor &in, Tensor &out);
    /// @}
    /// @name backward ops
    /// @{
    __HOST__ int    _bloss(Tensor &tgt);
    __HOST__ void   _bstep(Tensor &in, Tensor &out);
    __HOST__ int    _bconv(Tensor &in, Tensor &out);
    __HOST__ int    _blinear(Tensor &in, Tensor &out);
    __HOST__ int    _bactivate(Tensor &in, Tensor &out);
    __HOST__ int    _bpool(Tensor &in, Tensor &out, t4_layer fn);
    __HOST__ int    _bupsample(Tensor &in, Tensor &out, t4_layer fn);
    __HOST__ int    _bbatchnorm(Tensor &in, Tensor &out);
    /// @}
    /// @name debug functions
    /// @{
    __HOST__ int    _check_nan(Tensor &t);
    __HOST__ void   _dump_f(const char *fn, Tensor &f);
    __HOST__ void   _dump_b(const char *bn, Tensor &b);
    __HOST__ void   _dump_w(const char *wn, Tensor &w, bool full=true);
    /// @}
};

} // namespace t4::nn

#endif // (!defined(__NN_MODEL_H) && T4_DO_OBJ && T4_DO_NN)
