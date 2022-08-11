# tensorForth - Release 2.2 / 2022-09
A Forth word can be seen as a nested function that process data flow, i.e. y = fn(fn-1(...(f2(f1(x))...))). We can use chain rule to collection derivations for forward and backward diff.
  
## Features
  * KV   new data structure to keep kv pair (i.e. associative array)
  * CNN
    + define word as the functional (implicit sequential)
    + words are destructive by default (i.e. input tensor updated)

### CNN Application Example
<pre>
: seq1 (N -- N') 0.5 10 conv2d 2 maxpool relu ;
: seq2 (N -- N') 0.5 20 conv2d 0.5 dropout 2 maxpool relu ;
: lin1 (N -- N') flatten relu 0.0 50 linear ;
: lin2 (N -- N') 0.5 dropout 0.0 10 linear ;
20 model 1 autograd                  \ create a network model of max 20 layers
1 28 28 1 tensor                     \ create an input tensor
>n                                   \ setup model input dimension
seq1 seq2 lin1 lin2 softmax loss.nll \ add layers to model
constant mnist              
: train (N set0 -- N') for_batch forward backprop next 0.1 0.9 sgd ;
: test  (N set1 -- N') for_batch forward avg predict . next ;
set0 train nn.save net_1             \ trainning session (and save the network)
set1 test                            \ testing session
nn.load net_1 set2 test              \ load network and test
</pre>

### Case Study - MNIST
|word|forward|param|network DAG|grad_fn|param[grad]|
|---|---|---|---|---|---|
|   |for_batch|(INs -- IN)        |IN                |        |                                  |
|seq1|conv2d  |(IN b1 c1 -- C1)   |IN [f1 b1 df1 db1]|dconv2d |(IN [f1 b1] dC1 -- dIN [df1 db1]) |
|    |relu    |(C1 -- R1)         |C1                |drelu   |(C1 dR1 -- dC1)                   |
|seq2|conv2d  |(R1 b2 c2 -- C2)   |R1 [f2 b2 df2 db2]|dconv2d |(R1 [f2 b2] dC2 -- dR1 [df2 db2]) |
|    |relu    |(C2 -- R2)         |C2                |drelu   |(C2 dR2 -- dC2)                   |
|    |maxpool |(R2 fp -- PX)      |R2 [fp]           |dmaxpool|(R2 [fp] dPX -- dR2)              |
|lin1|flatten |(PX -- FC)         |PX [sz]           |dflatten|(PX [sz] dFC -- dPX)              |
|    |linear  |(FC b3 n3 -- Z)    |FC [w3 b3 dw3 db3]|dlinear |(FC [w3 b3] dZ -- dFC [dw3 db3])  |
|lin2|relu    |(Z  -- R3)         |Z                 |drelu   |(Z dR3 -- dZ)                     |
|    |linear  |(R3 b4 n4 -- OUT)  |R3 [w4 b4 dw3 db3]|dlinear |(R3 [w4 b4] dOUT -- dR3 [dw4 db4])|
|fwd |softmax |(OUT -- PB)        |PB labels         |-       |(PB labels -- dOUT)               |
|    |loss.ce |(PB labels -- loss)|                  |        |                                  |
||||||
|back|sgd     |(loss b1 b2 -- )   |[f1' b1' f2' b2' w3' b3' w4' b3']|||

### Study NN forward and backward propegation
* https://explained.ai/matrix-calculus/
* https://en.wikipedia.org/wiki/Automatic_differentiation
* https://en.wikipedia.org/wiki/Adept_(C%2B%2B_library) and Stan
* for 1D nn   https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
* for 2D conv https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
<pre>
def linear(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z, cache
    
def d_linear(dZ, cache):
    A_prev, W, b = cache
    m  = A_prev.shape[1]
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db
    
def sigmoid(x):                                 # Sigmoid  
    return 1 / (1 + exp(-x))
    
def d_sigmoid(x):                               # Sigmoid derivative  
    return sigmoid(x) * (1 - sigmoid(x))
    
def softmax(x):                                 # Softmax
    return np.exp(x) / np.sum(np.exp(x))

def d_softmax(x):                               # Softmax derivative
    I = np.eye(x.shape[0])
    return softmax(x) * (I - softmax(x).T)
    
def cross_e(y_true, y_pred):                    # CE
    return -sum(y_true * np.log(y_pred + 10**-100))

def d_cross_e(y_true, y_pred):                  # CE derivative
    return -y_true / (y_pred + 10**-100)
</pre>

### CNN volcabularies
#### Tensor ops
|word|param/example|tensor creation ops|
|---|---|---|
|stack|(Aa Ab i - Aa Ab Tc)|stack arrays on given axis|
|split|(Ta i - Ta Aa Ab Ac) |split matrix into matrix on a given axis|

#### Load/Save - .npy
|word|param/example|tensor creation ops|
|---|---|---|
|nn.dir|( -- )|list dataset directory|
|nn.load|( -- N)|load trained network|
|nn.save|(N -- )|export network as a file|
    
#### Convolution and Linear funtions (destructive by default)
|word|param/example|tensor creation ops|
|---|---|---|
|conv2d|(N -- N')|create a 2D convolution 3x3 filter, stride=1, padding=same, dilation=0, bias=0.5|
|conv2d|(N b c -- N')|create a 2D convolution, bias=b, c channels output, with default 3x3 filter|
|conv2d|(N b c A -- N')|create a 2D convolution, bias=b, c channels output, with config i.g. Vector[5, 5, 3, 2, 1] for (5x5, padding=3, stride=2, dilation=1, bais=0.3)|
|flatten|(N -- N')|flatten a tensor (usually input to linear)|
|linear|(N b n -- N')|linearize (y = Wx + b) from Ta input to n out_features|

#### Activation (non-linear)
|word|param/example|tensor creation ops|
|---|---|---|
|relu|(N -- N')|Rectified Linear Unit|
|tanh|(N -- N')|Tanh Unit|
|sigmoid|(N -- N')|1/(1+exp^-z)|
|softmax|(N -- N')|probability vector exp(x)/sum(exp(x))|
    
#### Pooling and Dropout (Downsampling)
|word|param/example|tensor creation ops|
|---|---|---|
|avgpool|(N n -- N')|nxn cells average pooling|
|maxpool|(N n -- N')|nxn cells maximum pooling|
|minpool|(N n -- N')|nxn cell minimum pooling|
|dropout|(n p -- N')|zero out p% of channel data (add noise between data points)|
  
#### Loss
|word|param/example|tensor creation ops|
|---|---|---|
|loss.nll|(N Ta -- N Ta')|negative likelihood|
|loss.mse|(N Ta -- N Ta')|mean sqare error|
|loss.ce|(N Ta -- N Ta')|cross-entropy|
|predict|(N -- N n)|cost function (avg all losts)|

#### Back Propergation
|word|param/example|tensor creation ops|
|---|---|---|
|autograd|(N n -- N')|enable/disable model autograd|
|backprop|(N Ta -- N')|stop DAG building, execute backward propergation|
|sgd|(N Ta p m -- N')|apply SGD(learn_rate=p, momentum=m) backprop on DAG|
|adam|(N Ta p m -- N')|apply Adam backprop|
    

