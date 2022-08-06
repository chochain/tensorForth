/**
 * @file
 * @brief tensorForth tensor class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENSOR_H_
#define TEN4_SRC_TENSOR_H_
#include <ostream>
#include "ten4_types.h"
#include "util.h"
//===============================================================================
/// tensorForth tensor class
/// @brief - Tensor at rank=4, row-major, F32 only storage
/// Note:
///    PyTorch.Tensor: size, dtype, type_id, stride, tensorstore
///
typedef enum {
    ADD = 0,
    SUB,
    MUL,
    DIV,
    ABS,
    EXP,
    TANH,
    RELU,
    FILL,
    SCALE
} t4_mat_op;

typedef enum {
    TENSOR = 0,            ///< tensor object
    VIEW,                  ///< a view object
    LAYER,                 ///< neural network layer
    ACTI                   ///< activation
} t4_obj;

struct  Tensor;            ///< forward declaration
typedef void (*GradFn)(Tensor&, Tensor&);

struct Tensor : public Managed {
    U32      size;         ///< number of data elements, TODO: more than 4G elements
    union {
        U32  attr = 0;     ///< attrbutes collective
        struct {
            U8     dsize;  ///< size of data element, F32 for now
            U8     rank;   ///< rank of tensor 2:matrix, 4:NHWC tensor
            U8     xxx;    ///< reserved
            t4_obj ttype;  ///< 0: tensor, 1: view, 2: layer, 3: activation
        };
    };
    U16      stride[4];    ///< strides to calculate memory offset
    U16      shape[4];     ///< shape=HWCN, matrix C=N=1, vector W=C=N=1
    U8       *data = 0;    ///< managed memory block pointer
    Tensor   *grad[4];     ///< gradiant and jacobian tensors
    GradFn   grad_fn = 0;  ///< grandiant funtion pointer
    ///
    /// static ops
    /// Note:
    ///   1. resultant tensor as last parameter
    ///   2. return the resultant tensor
    ///
    static __BOTH__ Tensor &gemm(Tensor &A, Tensor &B, Tensor &C, DU alpha, DU beta);
    static __BOTH__ Tensor &mm(Tensor &A, Tensor &B, Tensor &C) { return gemm(A, B, C, 1.0, 0.0); }
    static __BOTH__ Tensor &mat(t4_mat_op op, Tensor &A, Tensor &B, Tensor &C);  ///> matrix-matrix element-wise ops (Hadamard)
    static __BOTH__ Tensor &mat(t4_mat_op op, Tensor &A, DU v, Tensor &C);       ///> matrix-scalar element-wise ops
    static __BOTH__ Tensor &copy(Tensor &A, Tensor &C);
    static __BOTH__ Tensor &transpose(Tensor &A, Tensor &T);
    static __BOTH__ Tensor &inverse(Tensor &A, Tensor &I);  /// GaussJordan (with Pivot)
    static __BOTH__ Tensor &inverse(Tensor &LU);            /// from LU (no Pivot)
    static __BOTH__ Tensor &lu(Tensor &A);                  /// LU (no Pivot)
    static __BOTH__ Tensor &plu(Tensor &A, Tensor &P);      /// LU with permutation vector
    ///
    /// class contructors
    ///
    __HOST__ Tensor();
    __HOST__ Tensor(U16 n, U16 h, U16 w, U16 c);
    __HOST__ Tensor(U16 h, U16 w);
    __HOST__ Tensor(U32 sz);
    __HOST__ ~Tensor();
    ///
    /// attributes
    ///
    __BOTH__ __INLINE__ U16  N()       { return shape[3]; }
    __BOTH__ __INLINE__ U16  H()       { return shape[0]; }
    __BOTH__ __INLINE__ U16  W()       { return shape[1]; }
    __BOTH__ __INLINE__ U16  C()       { return shape[2]; }
    __BOTH__ __INLINE__ bool is_view() { return ttype == VIEW; }
    ///
    /// tensor arithmetics
    ///
    __BOTH__ DU     sum();
    __BOTH__ DU     dot(Tensor &B);
    __BOTH__ Tensor &map(t4_mat_op op, DU v=DU0); ///< element-wise absolute
    ///
    /// linear algebra methods
    ///
    __BOTH__ DU     det();                    ///< matrix determinant
    __BOTH__ Tensor &triu();                  ///< upper triangle
    __BOTH__ Tensor &tril();                  ///< lower triangle
    ///
    /// tensor life-cycle ops
    ///
    __BOTH__ Tensor &set_as_view(bool set=true);
    __BOTH__ Tensor &reset(void *mptr, U32 sz);
    __BOTH__ Tensor &reshape(U32 sz);
    __BOTH__ Tensor &reshape(U16 h, U16 w);
    __BOTH__ Tensor &reshape(U16 n, U16 h, U16 w, U16 c);
    __BOTH__ Tensor &identity();              ///< fill as an identity matrix
    __HOST__ void   copy_to_host(void* dst) { cudaMemcpy(dst, data, size, cudaMemcpyDeviceToHost); }
    ///
    /// IO
    ///
    __BOTH__ void to_s(std::ostream &fout);
    ///
    /// TODO: tensor arithmetics
    ///
    __BOTH__ __INLINE__ Tensor &operator+=(Tensor &t){ return *this; }
    __BOTH__ __INLINE__ Tensor &operator-=(Tensor &t){ return *this; }
    __BOTH__ __INLINE__ Tensor &operator+(Tensor &t) { return *this; }
    __BOTH__ __INLINE__ Tensor &operator-(Tensor &t) { return *this; }
    __BOTH__ __INLINE__ Tensor &operator*(Tensor &t) { return *this; }
    __BOTH__ __INLINE__ Tensor &operator/(Tensor &t) { return *this; }
    __BOTH__ __INLINE__ Tensor &operator%(Tensor &t) { return *this; }
    ///
    /// TODO: tensor logical ops
    ///
    __BOTH__ __INLINE__ bool   operator<(Tensor &t)  { return 0; }
    __BOTH__ __INLINE__ bool   operator>(Tensor &t)  { return 0; }
    __BOTH__ __INLINE__ bool   operator<=(Tensor &t) { return 0; }
    __BOTH__ __INLINE__ bool   operator>=(Tensor &t) { return 0; }
    __BOTH__ __INLINE__ bool   operator==(Tensor &t) { return 0; }
    __BOTH__ __INLINE__ bool   operator!=(Tensor &t) { return 0; }
};
#endif // TEN4_SRC_TENSOR_H_
