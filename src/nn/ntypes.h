/**
 * @file
 * @brief Tensor class - ranked tensor object interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __NN_NTYPES_H
#define __NN_NTYPES_H
#pragma once

#if (T4_DO_OBJ && T4_DO_NN)

namespace t4 {

//===============================================================================
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

#define LAYER_OP \
    "output ", "conv2d ", "linear ", "flatten", "relu   ", \
    "tanh   ", "sigmoid", "selu   ", "leakyrl", "elu    ", \
    "dropout", "softmax", "logsmax", "avgpool", "maxpool", \
    "minpool", "batchnm", "upsampl"

typedef enum {
    LOSS_MSE = 0,            ///< mean square error
    LOSS_BCE,                ///< binary cross entropy (sigmoid input)
    LOSS_CE,                 ///< cross entropy (softmax input)
    LOSS_NLL                 ///< negative log-likelihood (logsoftmax input)
} t4_loss;

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

} // namespace t4

#endif // T4_DO_NN
#endif // __NN_NTYPES_H
