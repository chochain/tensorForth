# tensorForth - Release 3.0 / 2022-09
A Forth word can be seen as a nested function that process data flow, i.e. y = fn(fn-1(...(f2(f1(x))...))). We can use chain rule to collection derivations for forward and backward diff.
  
## Features
  * Tensor Format support
    + NHWC (as in TensorFlow) supported
    + NCHW (as in PyTorch), future release
  * Neural Network Model
    + words are functional (implicit sequential)
    + layers: conv2d, linear, flatten
    + pooling: maxpool, minpool, avgpool, dropout
    + activation: relu, sigmoid
    + loss: ce, mse
    + feedforward (autograd)
    + backprop
    + gradiant: sgd, adam
  * add OpenGL viewer (for dataset)
  * models load/save - VM pause/resume

### CNN Application Example
<pre>
10 28 28 1 nn.model                       \ create a network model (input dimensions)
0.5 10 conv2d 2 maxpool relu              \ add a convolution block
0.5 20 conv2d 0.5 dropout 2 maxpool relu  \ add another convolution block
flatten 0.0 49 linear                     \ add reduction layer, and the
0.5 dropout 0.0 10 linear softmax         \ final fully connected output

constant md0                 \ we can store the model in a constant
                             \ now, define our training and testing flows
: my_train (N D -- N') nn.for forward loss.ce backprop 0.1 0.9 nn.sgd nn.next ;
: my_test  (N D -- N') nn.for forward loss.mse . batch nn.next ;

md0                          \ place the model on TOS
network                      \ optionally, display our 13-layer NN model
10 dataset mnist_train       \ create a dataset with batch_sz = 10 (from Loader repo)
my_train                     \ start training session
nn.save my_net               \ optinally, saving the trainned network

1 dataset mnist_test         \ create a test dataset batch_sz = 1
constant ds1                 \ save in a constant (or just using stack is OK)

md0 ds1 my_test              \ start test session directly, or
nn.load my_net ds1 my_test   \ load from trained network and test
</pre>

### Case Study - MNIST
|word|forward|param|network DAG|grad_fn|param[grad]|
|---|---|---|---|---|---|
|    |nn.for  |(INs -- IN)        |IN                |        |                                  |
|seq1|conv2d  |(IN b1 c1 -- C1)   |IN [f1 b1 df1 db1]|dconv2d |(IN [f1 b1] dC1 -- dIN [df1 db1]) |
|    |relu    |(C1 -- R1)         |C1                |drelu   |(C1 dR1 -- dC1)                   |
|seq2|conv2d  |(R1 b2 c2 -- C2)   |R1 [f2 b2 df2 db2]|dconv2d |(R1 [f2 b2] dC2 -- dR1 [df2 db2]) |
|    |relu    |(C2 -- R2)         |C2                |drelu   |(C2 dR2 -- dC2)                   |
|    |maxpool |(R2 fp -- PX)      |R2 [fp]           |dmaxpool|(R2 [fp] dPX -- dR2)              |
|lin1|flatten |(PX -- FC)         |PX [sz]           |dflatten|(PX [sz] dFC -- dPX)              |
|    |linear  |(FC b3 n3 -- Z)    |FC [w3 b3 dw3 db3]|dlinear |(FC [w3 b3] dZ -- dFC [dw3 db3])  |
|lin2|relu    |(Z  -- R3)         |Z                 |drelu   |(Z dR3 -- dZ)                     |
|    |linear  |(R3 b4 n4 -- R4)   |R3 [w4 b4 dw4 db4]|dlinear |(R3 [w4 b4] dR4 -- dR3 [dw4 db4]) |
|    |dropout |(R4 p -- OUT)      |R4 [p msk]        |ddropout|(R4 dOUT [p msk] -- dR4)          |
||||||
|fwd |softmax |(OUT -- PB)        |PB labels         |-       |(OUT labels -- dOUT)              |
|    |loss.ce |(PB labels -- loss)|                  |        |                                  |
||||||
|back|nn.sgd  |(N &eta; -- )      |[f1' b1' f2' b2' w3' b3' w4' b3']|f -= &eta; * df||

### Backpropagation Study - MNIST
{% include backprop.html %}

### Study NN forward and backward propagation
* https://explained.ai/matrix-calculus/
* https://github.com/dnouri/cuda-convnet
* https://en.wikipedia.org/wiki/Automatic_differentiation
* https://en.wikipedia.org/wiki/Adept_(C%2B%2B_library) and Stan
* for 1D nn   https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
* for 2D conv https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
* backprop    https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
* backprop    https://www.google.com/search?channel=fs&client=ubuntu&q=forward+backward+propagation
* code ref    https://github.com/rasmusbergpalm/DeepLearnToolbox

### CNN volcabularies
#### Tensor ops
|word|param/example|tensor creation ops|
|---|---|---|
|stack|(Aa Ab i - Aa Ab Tc)|stack arrays on given axis|
|split|(Ta i - Ta Aa Ab Ac)|split matrix into matrix on a given axis|

#### Load/Save - .npy
|word|param/example|tensor creation ops|
|---|---|---|
|nn.model|(n h w c -- N)|create a Neural Network model with (n,h,w,c) input|
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
|tanh|(Ta -- Ta')|tensor element-wise tanh Ta' = tanh(Ta)|
|relu|(Ta -- Ta')|tensor element-wise ReLU Ta' = max(0, Ta)|
|sigmoid|(Ta -- Ta')|tensor element-wise Sigmoid Ta' = sigmoid(Ta)|
|tanh|(N -- N')|add tanh layer to network model|
|relu|(N -- N')|add Rectified Linear Unit to network model|
|sigmoid|(N -- N')|add sigmoid 1/(1+exp^-z) activation to network model|
|softmax|(N -- N')|add probability vector exp(x)/sum(exp(x)) to network model, feeds loss.ce|
|logsoftmax|(N -- N')|add probability vector x - log(sum(exp(x))) to network model, feeds loss.nll|
    
#### Pooling and Dropout (Downsampling)
|word|param/example|tensor creation ops|
|---|---|---|
|maxpool|(N n -- N')|nxn cells maximum pooling|
|avgpool|(N n -- N')|nxn cells average pooling|
|minpool|(N n -- N')|nxn cell minimum pooling|
|dropout|(N p -- N')|zero out p% of channel data (add noise between data points)|
  
#### Loss
|word|param/example|tensor creation ops|
|---|---|---|
|loss.mse|(N Ta -- N Ta')|mean sqare error|
|loss.ce|(N Ta -- N Ta')|cross-entropy, takes output from softmax|
|loss.nll|(N Ta -- N Ta')|negative likelihood, takes output from logsoftmax|
|onehot|(N -- N Ta)|dataset one-hot vector|
|predict|(N -- N n)|cost function (avg all losts)|

#### Propagation controls
|word|param/example|tensor creation ops|
|---|---|---|
|nn.for|(N ds -- N')|loop through a data set|
|nn.next|(N ds -- N')|loop if any subset left|
|autograd|(N n -- N')|enable/disable model autograd|
|forward|(N in -- N')|execute one forward propagation, layer-by-layer in given model|
|backprop|(N out -- N')|execute one backward propagation, adding derivatives for all parameters|
|nn.sgd|(N Ta p m -- N')|apply SGD(learn_rate=p, momentum=m) backprop on DAG|
|nn.adam|(N Ta -- N')|apply Adam backprop (alpha=0.001, beta1=0.1, beta2=0.999, eps=1e-6)|
|nn.adam|(N Ta a b -- N')|apply Adam backprop with given alpha, beta1, (beta2=0.999, eps=1e-6)|
    

