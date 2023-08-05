# tensorForth - Release 3.0 / 2022-11
A Forth word can be seen as a nested function in processing data flow, i.e. y = fn(fn-1(...(f2(f1(x))))). On the foundation of tensor manipulation capabilities of Release 2.x, machine learning functions are build as simple Forth words in tensorForth. With the stack-centric operations, compared to popular frameworks like TensorFlow or PyTorch, tensorForth keeps data in GPU which reduce the typical input-output shuffle. The clean syntax of Forth makes this NN framework compact and hopefully easier to understand.
  
### A CNN Training Example - MNIST
<pre>
10 28 28 1 nn.model                       \ create a network model (input dimensions)
0.5 10 conv2d 2 maxpool relu              \ add a convolution block
0.5 20 conv2d 0.5 dropout 2 maxpool relu  \ add another convolution block
flatten 49 linear                         \ add reduction layer to 49 features, and the
0.5 dropout 10 linear softmax             \ final fully connected 10-feature output
constant md0                              \ we can store the model in a constant
                                
md0 batchsize dataset mnist_train         \ create a MNIST dataset with model batch size
constant ds0                              \ save dataset in a constant

variable acc 0 acc !                      \ create an accuracy counter, and zero it
variable lst 0 lst !
: stat cr .                               \ display statistics
  ." >" clock .
  ." : hit=" acc @ . 0 acc !
  ." , loss=" lst @ . cr ;
  
: epoch ( N D -- N' )                     \ an CNN epoch (thru entire dataset)
  for                                     \ loop thru dataset per mini-batch
    forward                               \ neural net forward pass
    loss.ce lst ! nn.hit acc +!           \ calculate loss and total hit rate
    backprop                              \ back propagate loss
    0.01 0.0 nn.sgd                       \ execute stochastic gradient decent
    46 emit                               \ display progress '.'
  next ;
: cnn ( N D n -- N' D )                   \ entire CNN in one word
  for epoch r@ stat ds0 rewind next ;     \ run multiple epochs

ds0                                       \ put dataset as TOS
19 cnn                                    \ execute multiple epochs
drop                                      \ drop dataset from TOS

s" tests/my_net.t4" nn.save               \ persist the trained network
</pre>

### A CNN Prediction Application Example
<pre>
1000 28 28 1 nn.model                     \ create a blank model
s" tests/my_net.t4" nn.load               \ load trained model and parameters from saved file

batchsize dataset mnist_test              \ create a test dataset with model batch size
constant ds1                              \ keep it in a constant

: bench for forward nn.hit . ds1 next ;   \ here we define our test system

ds1                                       \ put dataset on TOS
9 bench                                   \ start test sessions
</pre>

### Machine Learning Features
  * Model
    > + creation
    > + feed forward
    > + back propagation with autograd
    > + optimization
    > + persistence
  * Layers
    > + layers: conv2d, conv1x1, linear, flatten, upsample, batchnorm
    > + pooling: maxpool, minpool, avgpool, dropout
    > + activation: relu, tanh, sigmoid, selu, leakyrelu, elu, softmax, log_softmax
    > + loss: ce, mse, nll
  * Dataset
    > + format - NHWC (as in TensorFlow)
    > + mini-batch fetch
    > + rewind
    > + loader - MNIST
  * Viewer (some more work)
    > + OpenGL for dataset
    > + Output tensor in HWC raw format with util to convert to PNG

### Machine Learning vocabularies
#### Model creation and persistence
|word|param/example|tensor creation ops|
|---|---|---|
|nn.model|(n h w c -- N)|create a Neural Network model with (n,h,w,c) input|
|nn.load|(N adr len [fam] -- N')|load trained network from a given file name|
|nn.save|(N adr len [fam] -- N)|export network as a file|
    
#### Dataset ops
|word|param/example|tensor creation ops|
|---|---|---|
|dataset|(n -- D)|create a dataset with batch size = n, and given name i.e. 10 dataset abc|
|fetch|(D -- D')|fetch a mini-batch from dataset on return stack|
|rewind|(D -- D')|rewind dataset internal counters (for another epoch)|
|batchsize|(D -- D b)|get input batch size of a model|

#### Model Debug ops
|word|param/example|tensor creation ops|
|---|---|---|
|>n|(N T -- N')|manually add tensor to model|
|n@|(N n -- N T)|fetch layered tensor from model, -1 is the latest layer|
|network|(N -- N)|display network model|
|trainable|(N f -- N)|set/unset network model trainable flag|

#### Batch controls
|word|param/example|tensor creation ops|
|---|---|---|
|forward|(N -- N')|execute one forward path with rs[-1] dataset, layer-by-layer in given model|
|forward|(N ds -- N')|execute one forward propagation with TOS dataset, layer-by-layer in given model|
|backprop|(N -- N')|execute one backward propagation, adding derivatives for all parameters|
|backprop|(N T -- N')|execute one backward propagation with given onehot vector|
|for|(N ds -- N')|loop through a dataset, ds will be pushed onto return stack|
|next|(N -- N')|loop if any subset of dataset left, or ds is pop off return stack|

#### Convolution and Linear functions (destructive by default)
|word|param/example|tensor creation ops|
|---|---|---|
|conv1x1|(N b c -- N')|create a 1x1 convolution, bias=b, c channels output|
|conv2d|(N -- N')|create a 2D convolution 3x3 filter, stride=1, padding=same, dilation=0, bias=0.5|
|conv2d|(N b c -- N')|create a 2D convolution, bias=b, c channels output, with default 3x3 filter|
|conv2d|(N b c A -- N')|create a 2D convolution, bias=b, c channels output, with config i.g. Vector[5, 5, 3, 2, 1] for (5x5, padding=3, stride=2, dilation=1, bias=0.3)|
|flatten|(N -- N')|flatten a tensor (usually input to linear)|
|linear|(N b n -- N')|linearize (y = Wx + b) from Ta input to n out_features|
|linear|(N n -- N')|linearize (y = Wx) bias=0.0 from Ta input to n out_features|

#### Activation (non-linear)
|word|param/example|tensor creation ops|
|---|---|---|
|tanh|(Ta -- Ta')|tensor element-wise tanh Ta' = tanh(Ta)|
|relu|(Ta -- Ta')|tensor element-wise ReLU Ta' = max(0, Ta)|
|sigmoid|(Ta -- Ta')|tensor element-wise Sigmoid Ta' = sigmoid(Ta)|
|sqrt|(Ta -- Ta')|tensor element-wise Sqrt Ta' = sqrt(Ta)|
|relu|(N -- N')|add Rectified Linear Unit to network model|
|tanh|(N -- N')|add tanh layer to network model|
|sigmoid|(N -- N')|add sigmoid 1/(1+exp^-z) activation to network model, used in binary|
|selu|(N -- N')|add Selu layer to network model|
|leakyrelu|(N a -- N')|add leaky ReLU activation with slope=a to network model|
|elu|(N a -- N')|add exponential linear unit activation with alpha=a to network model|

#### Pooling, Dropout, and UpSampling
|word|param/example|tensor creation ops|
|---|---|---|
|maxpool|(N n -- N')|nxn cells maximum pooling|
|avgpool|(N n -- N')|nxn cells average pooling|
|minpool|(N n -- N')|nxn cell minimum pooling|
|dropout|(N p -- N')|zero out p% of channel data (add noise between data points)|
|upsample|(N n -- N')|upsample to nearest size=n, 2x2 and 3x3 supported|
|upsample|(N m n -- N')|upsample with method=m, size=n, 2x2 and 3x3 support|
|batchnorm|(N -- N')|add batchnorm layer with default momentum=0.1|
|batchnorm|(N m -- N')|add batchnorm layer with momentum=m|

#### Classifier
|word|param/example|tensor creation ops|
|---|---|---|
|softmax|(N -- N')|add probability vector exp(x)/sum(exp(x)) to network model, feeds loss.ce, used in multi-class|
|logsoftmax|(N -- N')|add probability vector x - log(sum(exp(x))) to network model, feeds loss.nll, used in multi-class|
    
#### Loss and hit count
|word|param/example|tensor creation ops|
|---|---|---|
|loss.mse|(N Ta -- N Ta n)|mean squared error, take output from linear layer|
|loss.bce|(N Ta -- N Ta n)|binary cross-entropy, takes output from sigmoid layer|
|loss.ce|(N Ta -- N Ta n)|cross-entropy, takes output from softmax activation|
|loss.nll|(N Ta -- N Ta n)|negative log likelihood, takes output from log-softmax activation|
|nn.loss|(N Ta -- N Ta n)|auto select loss function from last output layer|

#### Gradient ops
|word|param/example|tensor creation ops|
|---|---|---|
|nn.sgd|(N p -- N')|apply SGD(learn_rate=p, momentum=0.0) model back propagation|
|nn.sgd|(N p m -- N')|apply SGD(learn_rate=p, momentum=m) model back propagation|
|nn.adam|(N a b1 -- N')|apply Adam backprop alpha, beta1, default beta2=1-(1-b1)^3|
|nn.zero|(N -- N')|reset momentum tensors|
|nn.onehot|(N -- N T)|get cached onehot vector from a model|
|nn.hit|(N -- N n)|get number of hit (per mini-batch) of a model|

#### TODO: Tensor ops
|word|param/example|tensor creation ops|
|---|---|---|
|stack|(Aa Ab i - Aa Ab Tc)|stack arrays on given axis|
|split|(Ta i - Ta Aa Ab Ac)|split matrix into matrix on a given axis|

### Back-propagation Case Study - MNIST
{% include backprop.html %}
|word|forward|param|network DAG|grad_fn|param[grad]|
|---|---|---|---|---|---|
|    |for     |(INs -- IN)        |IN                |        |                                  |
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

### References for NN forward and backward propagation
* https://explained.ai/matrix-calculus/
* https://github.com/dnouri/cuda-convnet
* https://en.wikipedia.org/wiki/Automatic_differentiation
* https://en.wikipedia.org/wiki/Adept_(C%2B%2B_library) and Stan
* https://luniak.io/cuda-neural-network-implementation-part-1/

* Loss https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
* Binary Classification well explained https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c

* 1D nn   https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
* 2D conv https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html

* Backprop https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
* Backprop https://www.google.com/search?channel=fs&client=ubuntu&q=forward+backward+propagation

+ Batchnorm Backprop https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm
* BatchNorm Alternatives https://analyticsindiamag.com/alternatives-batch-normalization-deep-learning/

* GAN https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
* GAN applications https://github.com/nashory/gans-awesome-applications
  + Mnist Keras https://debuggercafe.com/vanilla-gan-pytorch/
  + Mnist Pytorch https://medium.com/intel-student-ambassadors/mnist-gan-detailed-step-by-step-explanation-implementation-in-code-ecc93b22dc60

+ Inception https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
+ ResNet https://d2l.ai/chapter_convolutional-modern/resnet.html

* code ref https://github.com/rasmusbergpalm/DeepLearnToolbox


