# tensorForth - Release 2.2 / 2022-09
A Forth word can be seen as a nested function that process data flow, i.e. y = fn(fn-1(...(f2(f1(x))...))). We can use chain rule to collection derivations for forward and backward diff.
  
## Features
  * KV   new data structure to keep kv pair (i.e. associative array)
  * CNN
    + define word as the functional (implicit sequential)
    + words are destructive by default (i.e. input tensor updated)

### CNN Application Example
<pre>
: seq1 (Ta -- Ta') 10 5 conv2d 2 maxpool2d relu ;
: seq2 (Ta -- Ta') 20 5 conv2d 0.5 dropout 2 maxpool2d relu ;
: lin1 (Ta -- Ta') -1 320 reshape relu 50 linear ;
: lin2 (Ta -- Ta') 0.5 dropout 10 linear ;
: fwd  (Ta -- Ta') seq1 seq2 lin1 lin2 softmax ;
: train (set0 -- ) autograd for_batch fwd loss.nll backprop next 0.1 0.9 sgd nn.save ;
: test  (set1 -- ) nn.load for_batch forward loss.nll avg predict next ;
set0 train
set1 test
</pre>

### Case Study - MNIST
|word|forward|param|grad_param|grad_fn|param[grad]|
|---|---|---|---|---|---|
|    |autograd|(   -- 0)  |0                 |        |                                  |
|seq1|conv2d  |(in -- c1) |in [f1 b1 df1 db1]|dconv2d |(in [f1 b1] dc1 -- dimg [df1 db1])|
|    |relu    |(c1 -- r1) |c1                |drelu   |(c1 dr1 -- dc1)                   |
|seq2|conv2d  |(r1 -- c2) |r1 [f2 b2 df2 db2]|dconv2d |(r1 [f2 b2] dc2 -- dr1 [df2 db2]) |
|    |relu    |(c2 -- r2) |c2                |drelu   |(c2 dr2 -- dc2)                   |
|    |maxpool |(r2 -- po) |r2 [fp]           |dmaxpool|(r2 [fp] dpo -- dr2)              |
|lin1|reshape |(po -- fc) |po [sz]           |dreshape|(po [sz] dfc -- dpo)              |
|    |linear  |(fc -- z)  |fc [w3 b3 dw3 db3]|dlinear |(fc [w3 b3] dz -- dfc [dw3 db3])  |
|lin2|relu    |(z  -- r3) |z                 |drelu   |(z dr3 -- dz)                     |
|    |linear  |(r3 -- out)|r3 [w4 b4 dw3 db3]|dlinear |(r3 [w4 b4] dout -- dr3 [dw4 db4])|
|fwd |softmax |(out -- pb)|pb labels         |-       |(pb labels -- dout)               |
|    |loss.ce |(pb labels -- loss)|          |        |                                  |
||||||
|back|sgd     |(loss b1 b2 -- )|[f1' b1' f2' b2' w3' b3' w4' b3']|||

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
|reshape|?|with negative value (i.e. inferred size = size / other dims)|

#### Load/Save - .npy
|word|param/example|tensor creation ops|
|---|---|---|
|nn.dir|( -- )|list dataset directory|
|nn.load|(root -- Ta Tb)|test_dataset train_dataset, i.g. Ta[10000,28,28,1], Tb[50000,28,28,1]|
|nn.save|(Ta -- )|save tensor/gradiant|
    
#### Conv2D (destructive by default)
|word|param/example|tensor creation ops|
|---|---|---|
|conv2d|(Ta -- Ta')|create a 2D convolution 3x3 filter, stride=1, padding=same, dilation=0, bias=0.5|
|conv2d|(Ta c n -- Ta')|create a 2D convolution, c channels output, with nxn filter|
|conv2d|(Ta c A -- Ta')|create a 2D convolution, c channels output, with config i.g. Vector[3, 3, 1, 0, 1, 0.3] for (3x3, stride=1, padding=0, dilation=1, bais=0.3)|

#### Activation (non-linear)
|word|param/example|tensor creation ops|
|---|---|---|
|relu|(Ta -- Ta')|Rectified Linear Unit|
|tanh|(Ta -- Ta')|Tanh Unit|
|sigmoid|(Ta -- Ta')|1/(1+exp^-z)|
|softmax|(Ta -- Ta')|probability vector exp(x)/sum(exp(x))|
    
#### Pooling (Downsampling)
|word|param/example|tensor creation ops|
|---|---|---|
|avgpool2d|(Ta n -- Ta')|nxn cells average pooling|
|maxpool2d|(Ta n -- Ta')|nxn cells maximum pooling|
|minpool2d|(Ta n -- Ta')|nxn cell minimum pooling|
  
#### Linear (fully connected)
|word|param/example|tensor creation ops|
|---|---|---|
|linear|(Ta n -- Ta')|linearize (y = Wx + b) from Ta input to n out_features|

#### Dropout (reduce strong correlation, improve feature independence)
|word|param/example|tensor creation ops|
|---|---|---|
|dropout|(Ta p -- Ta')|zero out p% of channel data (add noise between data points)|

#### Loss
|word|param/example|tensor creation ops|
|---|---|---|
|loss.nll|(Ta Tb -- Tc)|negative likelihood|
|loss.mse|(Ta Tb -- Tc)|mean sqare error|
|loss.ce|(Ta Tb -- Tc)|cross-entropy|
|predict|(Ta    -- n)|cost function (avg all losts)|

#### Back Propergation
|word|param/example|tensor creation ops|
|---|---|---|
|autograd|( -- )|enable Direct Acylic Graph building with zero gradiant|
|backprop|(Ga -- Ga Ta)|stop DAG building, execute backward propergation|
|sgd|(Ga Ta p m -- )|apply SGD(learn_rate=p, momentum=m) backprop on DAG|
|adam|(Ga Ta p m -- )|apply Adam backprop|
    

