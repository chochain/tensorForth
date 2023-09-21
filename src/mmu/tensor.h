/**
 * @file
 * @brief Tensor class - ranked tensor object interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENSOR_H
#define TEN4_SRC_TENSOR_H
#include "util.h"           // in ../mmu
#include "t4base.h"         // in ../mmu

#if T4_ENABLE_OBJ
//===============================================================================
/// tensorForth tensor class
/// @brief - Tensor at rank=4, row-major, F32 only storage
/// Note:
///    PyTorch.Tensor: size, dtype, type_id, stride, tensorstore
///
typedef enum {
    T_DOT = 0,
    T_SOLV,
    T_INV,
    T_LU,
    T_LUINV,
    T_DET,
    T_TRIU,
    T_TRIL,
    T_XPOS
} t4_ten_op;
#define TENSOR_OP "dot","solv","inv","lu","luinv","det","triu","tril","xpos"

typedef enum {
    L_NONE = 0,
    L_CONV,
    L_LINEAR,
    L_FLATTEN,
    L_RELU,         //> Rectified Linear Unit
    L_TANH,
    L_SIGMOID,
    L_SELU,         //> Scaled Exponential Linear Unit
    L_LEAKYRL,      //> Leaky ReLU
    L_ELU,          //> Exponential Linear Unit
    L_DROPOUT,
    L_SOFTMAX,
    L_LOGSMAX,
    L_AVGPOOL,
    L_MAXPOOL,
    L_MINPOOL,
    L_BATCHNM,      //> Batch Norm
    L_USAMPLE       //> UpSample
} t4_layer;

#define T4_LAYER_LIST \
    "output ", "conv2d ", "linear ", "flatten", "relu   ", \
    "tanh   ", "sigmoid", "selu   ", "leakyrl", "elu    ", \
    "dropout", "softmax", "logsmax", "avgpool", "maxpool", \
    "minpool", "batchnm", "upsampl"

typedef enum {
    MM_NONE  = 0,
    MM_INC   = 1,
    MM_A_TXP = 2,
    MM_B_TXP = 4
} t4_mm_opt;

typedef enum {
    LOSS_MSE = 0,            ///< mean square error
    LOSS_BCE,                ///< binary cross entropy (sigmoid input)
    LOSS_CE,                 ///< cross entropy (softmax input)
    LOSS_NLL                 ///< negative log-likelihood (logsoftmax input)
} t4_loss;

struct Tensor : public T4Base {
    U16      stride[4] = {1,1,1,1}; ///< stride=HWCN, for calc memory offset
    U16      shape[4]  = {1,1,1,1}; ///< shape=HWCN, matrix C=N=1, vector W=C=N=1
    t4_layer grad_fn   = L_NONE;    ///< grandiant funtion type
    Tensor   *grad[4];              ///< gradient and jacobian tensors
    Tensor   *mtum[4];              ///< momentum and delta tensors
    ///
    /// static ops
    /// Note:
    ///   1. resultant tensor as last parameter
    ///   2. return the resultant tensor
    ///
    static __GPU__  Tensor &ten_op(math_op op, Tensor &A, DU v, Tensor &O);       ///> matrix-scalar element-wise ops
    static __GPU__  Tensor &ten_op(math_op op, Tensor &A, Tensor &B, Tensor &O);  ///> matrix-matrix element-wise ops (Hadamard)
    static __GPU__  Tensor &mm(Tensor &A, Tensor &B, Tensor &O, t4_mm_opt opt=MM_NONE);
    static __GPU__  Tensor &gemm(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta);
    static __GPU__  Tensor &copy(Tensor &A, Tensor &O);
    static __GPU__  Tensor &transpose(Tensor &A, Tensor &T);
    static __GPU__  Tensor &inverse(Tensor &A, Tensor &I);  /// GaussJordan (with Pivot)
    static __GPU__  Tensor &lu(Tensor &A);                  /// LU (no Pivot)
    static __GPU__  Tensor &lu_inverse(Tensor &LU);         /// inverse a pre-processed LU (no Pivot)
    static __GPU__  Tensor &plu(Tensor &A, Tensor &P, int *ns);/// LU with permutation vector
    ///
    /// class contructors
    ///
    __HOST__ Tensor()       : T4Base() {}
    __HOST__ Tensor(U32 sz) : T4Base(sz) {
        H() = (U16)sz;
        WARN("vector[%d] allocated\n", numel);
    }
    __HOST__ Tensor(U16 h, U16 w) : T4Base(h, w) {
        H() = h; W() = w;
        WARN("matrix(%d,%d) allocated\n", h, w);
    }
    __HOST__ Tensor(U16 n, U16 h, U16 w, U16 c) : T4Base(n, h, w, c) {
        H() = h; W() = w; C() = c; N() = n;
        WARN("tensor(%d,%d,%d,%d) allocated\n", n, h, w, c);
    }
    __HOST__ ~Tensor() {
        switch (rank) {
        case 2: WARN("matrix(%d,%d) freed\n", H(), W()); break;
        case 4: WARN("tensor(%d,%d,%d,%d) freed\n", N(), H(), W(), C()); break;
        default: WARN("~Tensor error: rank=%d\n", rank);
        }
    }
    ///
    /// attributes
    ///
    __BOTH__ __INLINE__ U16  &N()  { return shape[3]; }
    __BOTH__ __INLINE__ U16  &H()  { return shape[0]; }
    __BOTH__ __INLINE__ U16  &W()  { return shape[1]; }
    __BOTH__ __INLINE__ U16  &C()  { return shape[2]; }
    __BOTH__ __INLINE__ U32  HWC() { return shape[0] * shape[1] * shape[2]; }
    __BOTH__ __INLINE__ DU   *slice(int n) { return &data[ n * HWC() ]; }
    __BOTH__ __INLINE__ bool is_same_shape(Tensor &t) {
#ifdef __CUDA_ARCH__
        return MEMCMP(shape, t.shape, sizeof(shape)) == 0;
#else  // __CUDA_ARCH
        return memcmp(shape, t.shape, sizeof(shape)) == 0;
#endif // __CUDA_ARCH__
    }
    ///
    /// tensor arithmetics
    ///
    __GPU__  DU     sum();
    __GPU__  DU     avg();                    ///< mean
    __GPU__  DU     std();                    ///< population standard deviation
    __GPU__  DU     max();
    __GPU__  DU     min();
    __GPU__  DU     dot(Tensor &B);
    __GPU__  DU     loss(t4_loss op, Tensor &tgt);
    ///
    /// linear algebra methods
    ///
    __GPU__  DU     det();                    ///< matrix determinant
    __GPU__  Tensor &triu();                  ///< upper triangle
    __GPU__  Tensor &tril();                  ///< lower triangle
    ///
    /// tensor life-cycle ops
    ///
    __BOTH__ Tensor &reset(void *mptr, U32 sz, t4_obj tt=T4_TENSOR, t4_layer fn=L_NONE);
    __BOTH__ Tensor &reshape(U32 sz);
    __BOTH__ Tensor &reshape(U16 h, U16 w);
    __BOTH__ Tensor &reshape(U16 n, U16 h, U16 w, U16 c);
    __BOTH__ Tensor &reshape(U16 c1, U16 n, U16 h, U16 w, U16 c);
    
    __BOTH__ Tensor &identity();                  ///< fill as an identity matrix
    __BOTH__ Tensor &map(math_op op, DU v=DU0); ///< element-wise absolute
    __BOTH__ Tensor &fill(DU v) { return this->map(FILL, v); }
    __BOTH__ Tensor &normalize(DU avg, DU std);
    __HOST__ void   copy_to_host(void* dst) { cudaMemcpy(dst, data, numel, cudaMemcpyDeviceToHost); }
    ///
    /// IO
    ///
    __BOTH__ void to_s(std::ostream &fout);
    ///
    /// tensor debugger
    ///
    static __BOTH__ void _dump(DU *v, int H, int W, int C);
    static __BOTH__ void _view(DU *v, int H, int W, int C, DU mean, DU scale);
    __GPU__ void show(bool dump=false);
    ///
    /// tensor-scalar operators
    ///
    __GPU__ __INLINE__ Tensor &operator=(DU v)      { return fill(v);     }
    __GPU__ __INLINE__ Tensor &operator+=(DU v)     { return map(ADD, v); }
    __GPU__ __INLINE__ Tensor &operator-=(DU v)     { return map(SUB, v); }
    __GPU__ __INLINE__ Tensor &operator*=(DU v)     { return map(MUL, v); }
    ///
    /// tensor-tensor arithmetic operators
    ///
    __GPU__ __INLINE__ Tensor &operator=(Tensor &t) { copy(t, *this); return *this; }
    __GPU__ __INLINE__ Tensor &operator+=(Tensor &t){ return ten_op(ADD, *this, t, *this); }
    __GPU__ __INLINE__ Tensor &operator-=(Tensor &t){ return ten_op(SUB, *this, t, *this); }
    __GPU__ __INLINE__ Tensor &operator*=(Tensor &t){ return ten_op(MUL, *this, t, *this); }
    ///
    /// tensor-tensor logical ops
    ///
    __GPU__ __INLINE__ bool   operator<(Tensor &t)  { return 0; }
    __GPU__ __INLINE__ bool   operator>(Tensor &t)  { return 0; }
    __GPU__ __INLINE__ bool   operator<=(Tensor &t) { return 0; }
    __GPU__ __INLINE__ bool   operator>=(Tensor &t) { return 0; }
    __GPU__ __INLINE__ bool   operator!=(Tensor &t) { return (intptr_t)this!=(intptr_t)&t; }
    __GPU__ __INLINE__ bool   operator==(Tensor &t) { return (intptr_t)this==(intptr_t)&t; }
};

#endif // T4_ENABLE_OBJ
#endif // TEN4_SRC_TENSOR_H
