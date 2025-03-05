/**
 * @file
 * @brief Model class - NN model (i.e. Container in PyTorch)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (!defined(__NN_MODEL_H) && T4_DO_OBJ && T4_DO_NN)
#define __NN_MODEL_H
#include "sys.h"             /// * ms, rand
#include "mmu/mmu.h"

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
    DU *parm, Tensor *w, Tensor *dw, Tensor *m, Tensor *v);
///
///< Neural Network Model class
///
class Model : public T4Base {
    MMU    *_mmu;              ///< memory controller
    Tensor *_store;            ///< model storage - Sequential, TODO: DAG
    Tensor *_hot  = NULL;      ///< cached dataset one-hot vector
    int    _hit   = 0;         ///< hit counter
    int    _iter  = 0;         ///< iteration counter (for Adam)
    int    _trace = 0;
    
public:
    int    epoch  = 0;         ///< TODO: for learning rate decay
    ///
    /// @name Derivertive ops
    /// @{
    static __HOST__ const char* nname(int n);    ///< network layer name on host
    static __GPU__  const char* d_nname(int n);  ///< network layer name on device
    /// @}
    /// @name layer access methods
    /// @{
    __BOTH__ Tensor &operator[](S64 i);          ///< 64-bit indexing (negative possible)
    __BOTH__ int    slots();
    __GPU__  void   reset(MMU *mmu, Tensor &store);
    __GPU__  Model  &npush(DU v);
    __GPU__  Model  &npush(Tensor &t);
    __GPU__  DU     npop();
    __GPU__  int    batch_size();
    /// @}
    /// @name Tensor constructors and randomizer
    /// @{
    __GPU__ Tensor &COPY(Tensor &t);                    ///< hardcopy a tensor (proxy to mmu)
    __GPU__ void   FREE(Tensor &t);
    __GPU__ Tensor &VEC(U64 sz);                        ///< proxy to MMU::tensor
    __GPU__ Tensor &T4(U32 n, U32 h);
    __GPU__ Tensor &T4(U32 n, U32 h, U32 w, U32 c);
    __GPU__ void   RAND(Tensor &t, DU scale);           ///< proxy to System::rand
    /// @}
    /// @name main NN methods
    /// @{
    __GPU__ Model  &add(t4_layer fn, U32 n=0, DU alpha=DU0, U16 *opt=NULL);
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
    /// @name gradient descent functions
    /// @{
    __GPU__ Model  &grad_zero() { _iter = _hit = 0; return *this; }
    __GPU__ Model  &grad_alloc(t4_optimizer op);        ///< allocate gradient vectors
    __GPU__ Model  &gradient(const char *nm,            ///< gradient descent functor
                             GdFunc fn,                 
                             DU *parm,
                             t4_optimizer op=OPTI_SGD); 
    __GPU__ Model  &sgd(DU lr, DU b);                   ///< stochastic gradient descent
    __GPU__ Model  &adam(DU lr, DU b1, DU b2);          ///< Adam gradient descent
    /// @}
    
private:
    /// @name Convolution and Linear initializer
    /// @{
    __GPU__ void   _iconv(Tensor &in, U32 c, DU bias, U16 *opt);    ///< 2D convolution
    __GPU__ void   _ilinear(Tensor &in, U32 n, DU bias);            ///< linearize (Dense) with n output
    __GPU__ void   _iflatten(Tensor &in);                           ///< flatten
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ void   _icopy(Tensor &in, t4_layer fn);                 ///< for softmax, logsoftmax
    __GPU__ void   _iactivate(Tensor &in, DU alpha, t4_layer fn );  ///< relu, tanh, sigmoid, zero out p% of channel data (add noise between data points)
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __GPU__ void   _ipool(Tensor &in, U16 f, t4_layer fn);          ///< pooling with nxn filter
    __GPU__ void   _ibatchnorm(Tensor &in, DU m);   ///< batch norm with momentum=m
    __GPU__ void   _iup(Tensor &in, U16 f, DU m);   ///< upsample with nxn filter
    /// @}
    /// @name forward ops
    /// @{
    __GPU__ void   _fstep(Tensor &in, Tensor &out);
    __GPU__ int    _fconv(Tensor &in, Tensor &out);
    __GPU__ int    _flinear(Tensor &in, Tensor &out);
    __GPU__ int    _factivate(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _fpool(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _fsoftmax(Tensor &in, Tensor &out);
    __GPU__ int    _flogsoftmax(Tensor &in, Tensor &out);
    __GPU__ int    _fbatchnorm(Tensor &in, Tensor &out);
    __GPU__ int    _fupsample(Tensor &in, Tensor &out);
    /// @}
    /// @name backward ops
    /// @{
    __GPU__ int    _bloss(Tensor &tgt);
    __GPU__ void   _bstep(Tensor &in, Tensor &out);
    __GPU__ int    _bconv(Tensor &in, Tensor &out);
    __GPU__ int    _blinear(Tensor &in, Tensor &out);
    __GPU__ int    _bactivate(Tensor &in, Tensor &out);
    __GPU__ int    _bpool(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _bupsample(Tensor &in, Tensor &out, t4_layer fn);
    __GPU__ int    _bbatchnorm(Tensor &in, Tensor &out);
    /// @}
    /// @name debug functions
    /// @{
    __GPU__ void   _dump_dbdf(Tensor &db, Tensor &df);
    __GPU__ void   _dump_db(Tensor &db);
    __GPU__ void   _dump_dw(Tensor &dw, bool full=true);
    /// @}
};
#endif // (!defined(__NN_MODEL_H) && T4_DO_OBJ && T4_DO_NN)
