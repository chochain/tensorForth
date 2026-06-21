<META HTTP-EQUIV='Content-Security-Policy' CONTENT="default-src 'self' ; script-src 'self' 'unsafe-inline' *.disqus.com a.disquscdn.com requirejs.org www.google-analytics.com; style-src 'self' 'unsafe-inline' a.disquscdn.com; img-src 'self' *; media-src 'self' ; frame-src disqus.com;">

## tensorForth - lives in GPU, does linear algebra and machine learning
* Forth VM that supports tensor calculus and Convolution Neural Network with dynamic parallelism in CUDA

### Status
* **CUDA12+** current, for Turing, Ampere, and on

|version|feature|stage|description|conceptual comparable|
|---|---|---|---|---|
|4.4|**Retentive**|analyzing|add RetNet ops|PyTorch.RetNet|
|5.0|**Transformer**|developing|add Transformer ops|PyTorch.Transformer|
|[4.0](https://github.com/chochain/tensorForth/releases/tag/v4.0.0)|**GAN + TB**|beta|+ TensorBoard output|PyTorch.GAN|

* **CUDA11.4** legacy version, for Kepler, Maxwell (i.e. Jetson Nano/TX), Pascal, and Volta only

|version|feature|stage|description|conceptual comparable|
|---|---|---|---|---|
|[3.2](https://github.com/chochain/tensorForth/releases/tag/v3.2.0)|**GAN**|beta|+ Generative Adversarial Net|PyTorch.GAN|
|[3.0](https://github.com/chochain/tensorForth/releases/tag/v3.0.0)|**CNN**|production|+ Machine Learning with autograd|Torch|
|[2.2](https://github.com/chochain/tensorForth/releases/tag/v2.2.2)|**lapack**|production|+ linear algebra methods|SciPy|
|[2.0](https://github.com/chochain/tensorForth/releases/tag/v2.0.2)|**matrix**|production|+ vector and matrix objects|NumPy|
|[1.0](https://github.com/chochain/tensorForth/releases/tag/v1.0.2)|**float**|production|extended eForth with F32 float|Python|

### What?
More details later, but here are some samples of tensorForth in action

* Machine Learning with Forth, shows progress on TensorBoard
  > |Generative Adversarial Network|TensorBoard Model Graph|
  > |---|---|
  > |<img src="https://raw.githubusercontent.com/chochain/tensorForth/master/docs/img/t4_gan_mnist_snip_all.png" width="600px">|<img src="https://raw.githubusercontent.com/chochain/tensorForth/master/docs/img/t4_tb_snip04.png" width="600px">|
  
* Build models with built-in layers, activations, and gradient descent methods
  > |Neural Network Models|Gradient Descent Methods|
  > |---|---|
  > |<img src="https://raw.githubusercontent.com/chochain/tensorForth/master/docs/img/ten4_model_cmp.png" width="600px" height="400px">|<img src="https://raw.githubusercontent.com/chochain/tensorForth/master/docs/img/ten4_gradient_cmp.png" width="600px" height="400px">|
  
  > |2D Convolution vs Linear+BatchNorm|Effectiveness of Different Activations|
  > |---|---|
  > |<img src="https://raw.githubusercontent.com/chochain/tensorForth/master/docs/img/ten4_cnv_vs_bn.png" width="600px" height="400px">|<img src="https://raw.githubusercontent.com/chochain/tensorForth/master/docs/img/ten4_act_cmp.png" width="600px" height="400px">|

### Why?
Compiled programs run fast on Linux. Command-line interface and shell scripting tie them together. Small tools are built along the way, productivity grows with time, especially in the hands of researchers.

*Niklaus Wirth*: **Algorithms + Data Structures = Programs**
* but too much on Algorithms - most modern languages, i.e. OOP, abstraction, template, ...
* or too focused on Data Structures - APL, SQL, ...

*Numpy* kind of solves both. So, for AI projects today, we use *Python* mostly. However, when GPU got involved, to enable processing on CUDA device, say with *Numba*, *TaiChi* or the likes, mostly there will be a behind the scene 'just-in-time' transcoding to C/C++ followed by compilation then load and run. In a sense, your *Python* code behaves like a *Makefile* which requires compilers/linker available on the host box. The common code-compile-run-debug cycle is especially counter-productive with ML's extra-long run stage.

Forth language encourages incremental build-test cycle. Having a 'GPU shell', that can interactively and incrementally develop/run each AI layer/node can potentially build a cleaner system. So, here we are!

> **tensor + Forth = tensorForth!**

### How?
* GPU, behaves like a co-processor or a DSP chip. It has no OS, no string support, and runs its own memory. Most of the available libraries are built for host instead of device i.e. to initiate calls from CPU into GPU but not the other way around. So, to be interactive, a memory manager, IO, and syncing with CPU are things needed to be had. It's pretty much like creating a Forth from scratch for a new processor as in the old days. CUDA Dynamic Parallelism was a perfect fit for the Forth VM running on a GPU and I had the entire REPL run within GPU without even coming back to host.
* **Post CUDA12**, unfortunately, nVidia has decided the cost of keeping track of internal synchronization was too high. Well, understanbly, the GPU cores became asynchronous. The v2.0 Dynamic Parallelism is not backward compatible. That sort of killed my dream of having everything on GPU! So, today, the architecture of **tensorForth** has VMs run in host and send computation-heavy tasks to GPU.
    
Since GPUs have good compiler support nowadays and I've ported the latest [*eForth*](https://github.com/chochain/eforth) to lambda-based in C++, pretty much all words can be transferred straight forward. With *FP32* or *float32* as the basic data unit, the addressing and logic ops took some attensions. Though today the codebase is in C++, the class/methods implementation might come back to Forth in the form of loadable blocks so it can be self-hosted.

It would be amusing to find someone brave enough to work the NVVM IR or even PTX assembly into a Forth that resides on GPU micro-cores in the fashion of [*GreenArray*](https://www.greenarraychips.com/), or to forge an FPGA doing similar kind of things.

In the end, languages don't really matter. It's the problem they solve. Having an interactive Forth in GPU does not mean a lot by itself. However, by adding matrix for linear algebra, or tensors for machine learning following the path from Numpy to PyTorch, with massively parallelism plus the cleanness of **Forth**, it might be useful one day, hopefully! 

### Example - Small Matrix ops
<pre>
> ten4                # enter tensorForth
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], vmss[64*1], pmem=48K, tensor=1024M
2 3 matrix{ 1 2 3 4 5 6 }            \ create a 2x3 matrix
 <0 T2[2,3]> ok                      \ 2-D tensor shown on top of stack (TOS)
dup                                  \ create a view of the matrix
 <0 T2[2,3] t[2,3]> ok               \ view shown in lower case
.                                    \ print the matrix (destructive as in Forth)
matrix[2,3] = {
    { +1.0000 +2.0000 +3.0000 }
    { +4.0000 +5.0000 +6.0000 } }
 <0 T2[2,3]> ok                      \ original matrix still on TOS
3 2 matrix ones                      \ create a 3x2 matrix, fill it with ones
 <0 T2[2,3] T2[3,2]> ok              \ now we have two matrices on stack
dup .                                \ see whether it is filled with one indeed
matrix[3,2] = {
    { +1.0000 +1.0000 }
    { +1.0000 +1.0000 }
    { +1.0000 +1.0000 } }
@                                    \ multiply them 2x3 @ 3x2
 <0 T2[2,3] T2[3,2] T2[2,2]> ok      \ 2x2 resultant matrix shown on TOS
.                                    \ print the new matrix
matrix[2,2] = {
    { +6.0000 +6.0000 }
    { +15.0000 +15.0000 } }
 <0 T2[2,3] T2[3,2]> ok
bye                                  \ exit tensorForth
 <0> ok
tensorForth 2.0 done.
</pre>

### Example - Larger Matrix ops - benchmark 1024x2048 x 2048x512 matrices - 1000 loops
<pre>
1024 2048 matrix rand                \ create a 1024x2048 matrix with uniform random values
 <0 T2[1024,2048]> ok                
2048 512 matrix ones                 \ create another 2048x512 matrix filled with 1s
 <0 T2[1024,2048] T2[2048,512]> ok
@                                    \ multiply them and resultant matrix on TOS
 <0 T2[1024,2048] T2[2048,512] T2[1024,512]> ok
2048 /= .                            \ scale down and print the resultant 1024x512 matrix
matrix[1024,512] = {                 \ in PyTorch style (edgeitem=3)
    { +0.4873 +0.4873 +0.4873 ... +0.4873 +0.4873 +0.4873 }
    { +0.4274 +0.4274 +0.4274 ... +0.4274 +0.4274 +0.4274 }
    { +0.5043 +0.5043 +0.5043 ... +0.5043 +0.5043 +0.5043 }
    ...
    { +0.5041 +0.5041 +0.5041 ... +0.5041 +0.5041 +0.5041 }
    { +0.5007 +0.5007 +0.5007 ... +0.5007 +0.5007 +0.5007 }
    { +0.5269 +0.5269 +0.5269 ... +0.5269 +0.5269 +0.5269 } }
 <0 T2[1024,2048] T2[2048,512]> ok
: mx                                        \ now let's define a word 'mx' for benchmark loop
  clock >r                                  \ keep the init time (in msec) on return stack
  for @ drop next                           \ loops of matrix multiplication
  clock r> - ;                              \ time it (clock1 - clock0)
 <0 T2[1024,2048] T2[2048,512]> ok
999 mx                                      \ now try 1000 loops
 <0 T2[1024,2048] T2[2048,512] 3584> ok     \ that is 3.584 sec (i.e. ~3.6ms / loop)
</pre>

### Example - CNN Training on MNIST dataset
<pre>
10 constant N                               \ mini-batch sample count
N 28 28 1 nn.model                          \ create a network model (input dimensions)
0.5 10 conv2d 2 maxpool relu                \ add a convolution block
0.5 20 conv2d 0.5 dropout 2 maxpool relu    \ add another convolution block
flatten 49 linear                           \ add reduction layer to 49-feature, and
0.5 dropout 10 linear softmax               \ final 10-feature fully connected output
constant md0                                \ we can keep the model as a constant
                                
N dataset mnist_train                       \ create a MNIST dataset with model batch size
128 128 normalize                           \ adjust samples from [0,1) to [-1,1)
constant ds0                                \ keep the dataset as a constant

\ the entire CNN training framework here
: epoch ( M ds -- M' )                      \ one epoch thru entire training dataset
  for                                       \ loop thru dataset per mini-batch
    forward                                 \ neural network forward pass
    backprop                                \ neural network back propagation
    0.01 nn.sgd                             \ training with Stochastic Gradient Descent
  next ;                                    \ next mini-batch (kept on return stack)
: cnn ( M ds n -- M' ds ) 1-                \ run multiple epochs
  for epoch ds0 rewind next drop ;

ds0 20 cnn                                  \ put dataset as TOS, run the CNN for 20 epochs
s" tests/my_net.t4" save                    \ persist the trained network
</pre>

### To Build, and Verify
There are two versions of **tensorForth**. After nVidia moved to CUDA12, the Dynamic Parallelism of child-grid is now async. The CUDA11.4, synced version, can work only on older GPUs. Check the compability chart first [here](https://forums.developer.nvidia.com/t/ubuntu-install-specific-old-cuda-drivers-combo/214601/5)

Here's how to build

* on your OS, install nVidia 470 driver
* install Docker Engine on your box, follow standard Docker installation guide
* install nvidia-container-toolkit
* pull a CUDA 11.4 + Ubuntu 20.04 docker image, a template is provided in ~/examples/cuda114_Docker
* run the CUDA container with your environment variables, a template provided in ~/examples/docker_cuda114

    <pre>
    > # for my old GT1030 on Ubuntu 20.04
    > cd examples
    > cp cuda114_Dockerfile Dockerfile
    > docker build -t cuda:v114 .
    > ./docker_cuda114
    </pre>

* or install nVidia 535 driver
* install a CUDA 12.2 + Ubuntu 22.04 docker image, a template is provided in ~/examples/cuda122_Docker
* run the CUDA container with your environment variables, a template provided in ~/examples/docker_cuda122

    <pre>
    > # for my cheap GTX1660 on Ubuntu 22.04
    > cd examples
    > cp cuda122_Dockerfile Dockerfile
    > docker build -t cuda:v122 .
    > ./docker_cuda122
    </pre>

Note: Those cards are what I have. Let me know if it works on your cards/OS.

#### build with Makefile (Linux)

    cd to your ten4 repo directory,
    per your GPU card, update root Makefile to your desired CUDA_ARCH, CUDA_CODE
        * for example: GT1030 sm_61, GTX1660 sm_75
    type 'make -j4 all' to build

#### build using IDE (Eclipse)

    install Eclipse
    install CUDA SDK 11.4 or 12.2 for Eclipse (from Nvidia site)
    create project by importing from your local repo root
    exclude directories - ~/tests, ~/img
    set File=>Properties=>C/C++ Build=>Setting=>NVCC compiler
      + Dialect=C++17
      + CUDA=11.4 or 12.2 (depends on your download)
      + Optimization=O2 or O3
    build project

#### tensorForth command line options
If all goes well, some warnings aside, *~/tests/ten4* is your executable. The following command line options are available
<pre>
    \-h             - list all GPU id and their properties
    \-d device_id   - select GPU device id
    \-v verbo_level - set verbosity level 0: off (default), 1: mmu tracing on, 2: detailed trace
</pre>

### Verifcation Cases
* Test v1 eForth ops

    ~/tests> ./ten4 < ../examples/lesson_10a.txt # for basic syntax checks

    you should see lots of output including the following
    <pre>
    ... 
     2026 year 1 month => 
    *************************************************************
         sun     mon     tue     wed     thu     fri     sat
    *************************************************************
                                           1       2       3
           4       5       6       7       8       9      10
          11      12      13      14      15      16      17
          18      19      20      21      22      23      24
          25      26      27      28      29      30      31
    *************************************************************
    ...
    </pre>

* Test v2 matrix ops

    ~/tests> ./ten4 < ../examples/lesson_20a.txt # for matrix ops
    <pre>
    ...
    { { 6 6 } { 9 9 } } * { { 0.5 0.5 } { 0.5 0.5 } } -1 T2[2,2] -> ok
    verify { { 3 3 } { 4.5 4.5 } } => matrix[2,2] = {
        { +3.0000 +3.0000 }
        { +4.5000 +4.5000 } }
    ...
    </pre>

    ~/tests> ./ten4 < ../examples/lesson_22a.txt # for linear algebra stuffs
    <pre>
    ...
    verify { 1 1 1 } => vector[3] = { 1 1 1 }
    ...
    verify { 8 -1 -8 } => vector[3] = { 8 -1 -8 }
    ...
    </pre>

* Test v3 ML ops

    ~/tests> ./ten4 < ../examples/lesson_30a.txt # NN model - single pass forward
    <pre>
    ...
    verify { { 6 } { 13 } { 20 } } => tensor[1,3,1,1] = { {
        { +6.0000 }
        { +13.0000 }
        { +20.0000 } } }
    ...
    </pre>

    ~/tests> ./ten4 < ../examples/lesson_30b.txt # NN model - single pass loss, and backprop

    ~/tests> ./ten4 < ../examples/lesson_30c.txt # NN model - 2-sample full round-trip

    ~/tests> ./ten4 < ../examples/lesson_30d.txt # CNN - MNIST single pass
    <pre>
    ...
    NN model[13/128]
    [  1] conv2d :T4[2,10,10,1] #p=40 T4[1,3,3,2] T1[2] bias=0.5, C=2
    [  2] maxpool:T4[2,10,10,2] #p=0 n=2
    [  3] relu   :T4[2,5,5,2] #p=100 T4[2,5,5,2] 
    [  4] conv2d :T4[2,5,5,2] #p=76 T4[2,3,3,2] T1[2] bias=0.5, C=2
    [  5] dropout:T4[2,5,5,2] #p=100 T4[2,5,5,2] rate=50%
    [  6] maxpool:T4[2,5,5,2] #p=0 n=2
    [  7] relu   :T4[2,2,2,2] #p=16 T4[2,2,2,2] 
    [  8] flatten:T4[2,2,2,2] #p=0 
    [  9] linear :T4[2,8,1,1] #p=882 T4[1,49,8,1] T1[49] bias=0, H=49
    [ 10] dropout:T4[2,49,1,1] #p=98 T4[2,49,1,1] rate=50%
    [ 11] linear :T4[2,49,1,1] #p=400 T4[1,4,49,1] T1[4] bias=0, H=4
    [ 12] softmax:T4[2,4,1,1] #p=4 T4[1,4,1,1] 
    [ 13] output :T4[2,4,1,1] #p=0 
    ...
    Model#backprop starts {
      0.00:12> softmax [ 2, 4, 1, 1]    p= 0.000 <= out'Σ/n=  0.00 [ 2, 4, 1, 1]
      0.12:11> linear  [ 2,49, 1, 1]    p= 0.000 <= out'Σ/n=  0.00 [ 2, 4, 1, 1]
      0.25:10> dropout [ 2,49, 1, 1]    p= 0.500 <= out'Σ/n= -0.13 [ 2,49, 1, 1]
      0.12: 9> linear  [ 2, 8, 1, 1]    p= 0.000 <= out'Σ/n=  0.21 [ 2,49, 1, 1]
      0.25: 8> flatten [ 2, 2, 2, 2]    p= 0.000 <= out'Σ/n= -0.01 [ 2, 8, 1, 1]
      0.12: 7> relu    [ 2, 2, 2, 2]    p= 0.000 <= out'Σ/n= -0.01 [ 2, 2, 2, 2]
      0.12: 6> maxpool [ 2, 5, 5, 2]    p= 0.000 <= out'Σ/n= -0.01 [ 2, 2, 2, 2]
      0.12: 5> dropout [ 2, 5, 5, 2]    p= 0.500 <= out'Σ/n=  0.33 [ 2, 5, 5, 2]
      0.25: 4> conv2d  [ 2, 5, 5, 2]    p= 0.500 <= out'Σ/n=  0.33 [ 2, 5, 5, 2]
      0.12: 3> relu    [ 2, 5, 5, 2]    p= 0.000 <= out'Σ/n=  0.14 [ 2, 5, 5, 2]
      0.12: 2> maxpool [ 2,10,10, 2]    p= 0.000 <= out'Σ/n=  0.00 [ 2, 5, 5, 2]
      0.12: 1> conv2d  [ 2,10,10, 1]    p= 0.500 <= out'Σ/n=  0.00 [ 2,10,10, 2]
    } Model::backprop  1.88 ms
    ...
    </pre>

    ~/tests> ./ten4 < ../examples/lesson_30e.txt # CNN - MNIST full framework, 20 epochs

* Tests v3.2 GAN ops

    ~/tests> ./ten4 < ../examples/lesson_32a.txt # GAN on NN single sample linear 2x2 layer verify

    ~/tests> ./ten4 < ../examples/lesson_32b.txt # GAN on MINST dataset, 100 epochs

## Machine Learning vocabularies (see [doc3](./docs/v3_progress.md) for detail and examples)
### Model creation, query, and persistence
<pre>
  nn.model   ( n h w c -- M )      - create a Neural Network model with (n,h,w,c) input
  >n         ( M T -- M' )         - manually add tensor to model
  n@         ( M n -- M T )        - fetch layered tensor from model, -1 is the latest layer
  nn.w       ( M n -- M T )        - query weight tensor of nth layer (0 means N/A)
  nn.b       ( M n -- M T )        - query bias tensor of nth layer (0 means N/A)
  nn.dw      ( M n -- M T )        - query weight gradient tensor of nth layer (0 means N/A)
  nn.db      ( M n -- M T )        - query bias gradient tensor of nth layer (0 means N/A)
  nn.w=      ( M T n -- M' )       - set weight tensor of nth layer
  nn.b=      ( M T n -- M' )       - set bias tensor of nth layer
  network    ( M -- M )            - display network model

  load       ( M adr len [fam] -- M' ) - load trained network from a given file name
  save       ( M adr len [fam] -- M )  - export network as a file
</pre>

### Dataset and Batch controls
<pre>
  dataset    ( n -- D )            - create a dataset with batch size = n, and given name i.e. 10 dataset abc
  normalize  ( D m s -- D' )
  fetch      ( D -- D' )           - fetch a mini-batch from dataset on return stack
  rewind     ( D -- D' )           - rewind dataset internal counters (for another epoch)
  batchsize  ( D -- D b )          - get input batch size of a model
  nn.len     ( D -- D n )          - query total num of samples of dataset from corpus

  forward    ( M -- M )            - execute one forward path with rs[-1] dataset, layer-by-layer in given model
  forward    ( M D -- M )          - execute one forward propagation with TOS dataset, layer-by-layer in given model
  backprop   ( M -- M' )           - execute one backward propagation, adding derivatives for all parameters
  backprop   ( M T -- M' )         - execute one backward propagation with given onehot vector

  for        ( M D -- M )          - loop through a dataset, ds will be pushed onto return stack
  next       ( M -- M )            - loop if any subset of dataset left, or ds is pop off return stack
  trainable  ( M f -- M' )         - enable/disable network trainable flag
</pre>

### Convolution, Dense, and Pooling Layers
<pre>
  conv2d     ( M -- M' )           - create a 2D convolution 3x3 filter, stride=1, padding=same, dilation=0, bias=0.5
  conv2d     ( M b c -- M' )       - create a 2D convolution, bias=b, c channels output, with default 3x3 filter
  conv2d     ( M b c A -- M' )     - create a 2D convolution, bias=b, c channels output, with config i.g. Vector[5, 5, 3, 2, 1] for (5x5, padding=3, stride=2, dilation=1, bias=0.3)
  conv1x1    ( M b c -- M' )       - create a 1x1 convolution, bias=b, c channels output, stride=1, padding=same, dilation=0
  flatten    ( M -- M' )           - flatten a tensor (usually input to linear)

  linear     ( M b n -- M' )       - linearize (y = Wx + b) from Ta input to n out_features
  linear     ( M n -- M' )         - linearize (y = Wx), bias=0.0 from Ta input to n out_features

  maxpool    ( M n -- M' )         - nxn cells maximum pooling
  avgpool    ( M n -- M' )         - nxn cells average pooling
  minpool    ( M n -- M' )         - nxn cell minimum pooling
  dropout    ( M p -- M' )         - zero out p% of channel data (add noise between data points)
  upsample   ( M n -- M' )         - upsample to nearest size=n, 2x2 and 3x3 supported
  upsample   ( M m n -- M' )       - upsample size=n, 2x2 and 3x3 supported, method=[0 nearest, 1=linear, 2=bilinear, 3=cubic
  batchnorm  ( M -- M' )           - batch normal layer with default momentum=0.1
  batchnorm  ( M m -- M' )         - batch normal with momentum=m
</pre>

### Activation (non-linear) and Classifier
<pre>
  tanh       ( M -- M' )           - add tanh layer to network model
  relu       ( M -- M' )           - add Rectified Linear Unit to network model
  sigmoid    ( M -- M' )           - add sigmoid 1/(1+exp^-z) activation to network model, used in binary cross entropy
  selu       ( M -- M' )           - add Selu alpha(exp-1) activation to network model
  leakyrelu  ( M a -- M' )         - add leaky ReLU with slope=a
  leu        ( M a -- M' )         - add exponential linear unit alpha=a
  
  softmax    ( M -- M' )           - add probability vector exp(x)/sum(exp(x)) to network model, feeds loss.ce, used in multi-class
  logsoftmax ( M -- M' )           - add probability vector x - log(sum(exp(x))) to network model, feeds loss.nll, used in multi-class
</pre>

### Loss and Gradient ops
<pre>
  loss.mse   ( M Ta -- M Ta n )    - mean squared error, takes output from linear layer
  loss.bce   ( M Ta -- M Ta n )    - binary cross-entropy, takes output from sigmoid activation
  loss.ce    ( M Ta -- M Ta n )    - cross-entropy, takes output from softmax activation
  loss.nll   ( M Ta -- M Ta n )    - negative log likelihood, takes output from log-softmax activation
  
  nn.loss    ( M Ta -- M Ta n )    - auto select between mse, bce, ce, nll based on last model output layer
  nn.zero    ( M -- M' )           - manually zero gradient tensors
  nn.sgd     ( M p -- M' )         - apply SGD(learn_rate=p, momentum=0.0) model back propagation
  nn.sgd     ( M p m -- M' )       - apply SGD(learn_rate=p, momentum=m) model back propagation
  nn.adam    ( M a -- M' )         - apply Adam backprop alpha=a, default b1=0.9, b2=0.999
  nn.adam    ( M a b1 -- M' )      - apply Adam backprop alpha=a, beta1=b1, default beta2=0.999
  nn.adam    ( M a b1 b2 -- M' )   - apply Adam backprop alpha=a, beta1=b1, beta2=b2
  nn.zero    ( M -- M' )           - reset momentum tensors
  nn.onehot  ( M -- M T )          - get cached onehot vector from a model
  nn.hit     ( M -- M n )          - get number of hit (per mini-batch) of a model
</pre>

## Tensor Calculus vocabularies (see [doc2](./docs/v2_progress.md) for detail and examples)
### Tensor creation
<pre>
  vector    ( n       -- T1 )      - create a 1-D array and place on top of stack (TOS)
  matrix    ( h w     -- T2 )      - create 2-D matrix and place on TOS
  tensor    ( n h w c -- T4 )      - create a 4-D NHWC tensor on TOS
  vector{   ( n       -- T1 )      - create 1-D array from console stream
  matrix{   ( h w     -- T2 )      - create a 2-D matrix from console stream
  view      ( T       -- T V )     - create a view (shallow copy) of a tensor
  copy      ( T       -- T T' )    - duplicate (deep copy) a tensor on TOS
</pre>

### Duplication ops (reference creation)
<pre>
  dup       ( Ta    -- Ta Ta )    - create a reference of a tensor on TOS
  over      ( Ta Tb -- Ta Tb Ta ) - create a reference of the 2nd item (NOS)
  2dup      ( Ta Tb -- Ta Tb Ta Tb )
  2over     ( Ta Tb Tc Td -- Ta Tb Tc Td Ta Tb )
</pre>

### Tensor/View print (destructive as in Forth)
<pre>
  . (dot)   ( V -- )  - print a view of a tensor
  . (dot)   ( T -- )  - print a vector, matrix, or tensor
  . (dot)   ( M -- )  - print a neaural network model
</pre>

### Shape adjustment (change shape of original tensor or view)
<pre>
  flatten   ( T -- T1' )             - reshape a tensor or view to 1-D array
  reshape2  ( T h w -- T2' )         - reshape to a 2-D matrix view
  reshape4  ( T n h w c -- T4' )     - reshape to a 4-D NHWC tensor or view
  same_shape? ( Ta Tb -- Ta Tb T/F ) - check whether Ta and Tb are the same shape
</pre>

### Fill tensor with init values (data updated to original tensor)
<pre>
  zeros     ( T   -- T' )   - fill tensor with zeros
  ones      ( T   -- T' )   - fill tensor with ones
  fill      ( T n -- T' )   - fill tensor with number on TOS
  gradfill  ( T   -- T' )   - gradient fill elements from 0 to 1
  eye       ( T   -- T' )   - fill diag with 1 and other with 0
  rand      ( T   -- T' )   - fill tensor with uniform random numbers
  randn     ( T   -- T' )   - fill tensor with normal distribution random numbers
  ={        ( T   -- T' )   - fill tensor with console input from the first element
  ={        ( T n -- T' )   - fill tensor with console input starting at n'th element
</pre>

### Tensor slice and dice
<pre>
  dim       ( T -- T Td )   - tensor dimensions, Td is a vector[4] of { N, H, W, C }
  t@        ( T i -- T n )  - fetch ith element from a tensor (in NHWC order)
  t!        ( T i n -- T' ) - store n into ith element of a tensor (in NHWC order)
  slice     ( T i0 i1 j0 j1 -- T T' ) - numpy.slice[i0:i1,j0:j1,]
</pre>

### Tensor-scalar, tensor-tensor arithmetic (by default non-destructive)
<pre>
  +         ( Ta Tb -- Ta Tb Tc )  - tensor element-wise addition Tc = Ta + Tb
  +         ( Ta n  -- Ta n  Ta' ) - tensor-scalar addition (broadcast) Ta' = Ta + n
  +         ( n  Ta -- n  Ta Ta' ) - scalar-tensor addition (broadcast) Ta' = Ta + n
  -         ( Ta Tb -- Ta Tb Tc )  - tensor element-wise subtraction Tc = Ta - Tb
  -         ( Ta n  -- Ta n  Ta' ) - tensor-scalar subtraction (broadcast) Ta' = Ta - n
  -         ( n  Ta -- n  Ta Ta' ) - tensor-scalar subtraction (broadcast) Ta' = n - Ta
  @         ( Ta Tb -- Ta Tb Tc )  - matrix-matrix inner product Tc = Ta @ Tb, i.e. matmul
  @         ( Ta Ab -- Ta Ab Ac )  - matrix-vector inner product Ac = Ta @ Ab
  @         ( Aa Ab -- Aa Ab n )   - vector-vector inner product n = Aa @ Ab, i.e. dot
  *         ( Ta Tb -- Ta Tb Tc )  - tensor-tensor element-wise multiplication Tc = Ta * Tb
  *         ( Ta Ab -- Ta Ab Ta' ) - matrix-vector multiplication Ta' = Ta * column_vector(Ab)
  *         ( Ta n  -- Ta n  Ta' ) - tensor-scalar multiplication Ta' = n * Ta, i.e. scale up
  *         ( n  Ta -- n  Ta Ta' ) - scalar-tensor multiplication Ta' = n * Ta, i.e. scale up
  /         ( Ta Tb -- Ta Tb Tc )  - tensor-tensor element-wise divide Tc = Ta / Tb
  /         ( Ta n  -- Ta n  Ta' ) - tensor-scalar scale down Ta' = 1/n * Ta
  sum       ( T     -- T n )       - sum all elements of a tensor
  avg       ( T     -- T n )       - average of all elements of a tensor
  max       ( T     -- T n )       - max of all elements of a tensor
  min       ( T     -- T n )       - min of all elements of a tensor
</pre>

### Tensor element-wise math ops (destructive, as in Forth)
<pre>
  abs       ( T -- T' )   - absolute T' = abs(T)
  exp       ( T -- T' )   - exponential T' = exp(T)
  ln        ( T -- T' )   - natural log T' = ln(T)
  log       ( T -- T' )   - logrithm tanh T' = log(T)
  tanh      ( T -- T' )   - tanh T' = tanh(T)
  relu      ( T -- T' )   - ReLU T' = max(0, T)
  sigmoid   ( T -- T' )   - Sigmoid T' = 1/(1+exp(-T))
  sqrt      ( T -- T' )   - Square Root T' = sqrt(T)
  1/x       ( T -- T' )   - reciprocal T' = 1/T
  negate    ( T -- T' )   - negate   T' = -(T)
</pre>

### Tensor-tensor arithmetic (destructive, as in Forth)
<pre>
  +=        ( Ta Tb -- Tc )    - tensor element-wise addition Tc = Ta + Tb
  +=        ( Ta n  -- Ta' )   - tensor-scalar addition (broadcast) Ta' = Ta + n
  +=        ( n  Ta -- Ta' )   - scalar-tensor addition (broadcast) Ta' = Ta + n
  -=        ( Ta Tb -- Tc )    - tensor element-wise subtraction Tc = Ta - Tb
  -=        ( Ta n  -- Ta' )   - tensor-scalar subtraction (broadcast) Ta' = Ta - n
  -=        ( n  Ta -- Ta' )   - scalar-tensor subtraction (broadcast) Ta' = n - Ta
  @=        ( Ta Tb -- Tc )    - matrix-matrix inner product Tc = Ta @ Tb, i.e. matmul
  @=        ( Ta Ab -- Ac )    - matrix-vector inner product Ac = Ta @ Ab
  @=        ( Aa Ab -- Ac )    - vector-vector inner product n = Aa @ Ab, i.e. dot
  *=        ( Ta Tb -- Tc )    - matrix-matrix element-wise multiplication Tc = Ta * Tb
  *=        ( Ta Ab -- Ac' )   - matrix-vector multiplication Ac' = Ta * Ab
  *=        ( Ta n  -- Ta' )   - tensor-scalar multiplication Ta' = n * Ta
  *=        ( n  Ta -- Ta' )   - scalar-tensor multiplication Ta' = n * Ta
  /=        ( Ta Tb -- Tc )    - matrix-matrix element-wise Tc = Ta / Tb 
  /=        ( Ta n  -- Ta' )   - tensor-scalar scale down multiplication Ta' = 1/n * Ta
</pre>

### Tensor-Tensor loss functions (by default destructive, as in Forth)
<pre>
  loss.mse  ( Ta Tb -- Ta' )   - Mean Square Loss
  loss.bce  ( Ta Tb -- Ta' )   - Binary Cross Entropy Loss
  loss.ce   ( Ta Tb -- Ta' )   - Categorical Cross Entropy Loss
  loss.nll  ( Ta Tb -- Ta' )   - Negative Log Likelihood Loss
</pre>

### Linear Algebra (by default non-destructive)
<pre>
  matmul    ( Ta Tb -- Ta Tb Tc ) - matrix-matrix multiplication Tc = Ma @ Mb
  matdiv    ( Ta Tb -- Ta Tb Tc ) - matrix-matrix division Mc = Ma @ inverse(Mb)
  inverse   ( T     -- T Ti )     - matrix inversion (via Gauss-Jordan with Pivot)
  luinv     ( T     -- T Ti )     - matrix inversion (via PLU factorization)
  plu       ( T     -- T Tp Tlu ) - Ma => P and L\U
  upper     ( T     -- T Tu)      - upper triangle
  lower     ( T     -- T Tl)      - lower triangle with diag filled with 1s
  transpose ( T     -- T Tx')     - matrix transpose
  det       ( T     -- T d)       - matrix determinant (with PLU)
  solve     ( Tb Ta -- Tb Ta Tx)  - solve linear equation Ta @ Tx = Tb
  gemm      ( a b Ta Tb Tc -- a b Tb Tb Tc' ) - GEMM Tc' = a * Ta * Tb + b * Tc
</pre>

### Tensor I/O, Persistence
<pre>
  save      ( T adr len [fam] -- T ) - pickle tensor to OS file (default text mode)
</pre>

### TODO - by priorities
* Refactor
  + host/kernel code separation 
  + study Scikit-learn (discrete functions)
  + study [Taichi](https://github.com/taichi-dev/taichi)
    - SNode
    - JIT
    - parallelization
    - auto diff
  + study JAX
    - JIT (XLA)
    - auto parallelization (pmap)
    - auto vectorization (vmap)
    - auto diff (grad), diffrax (RK4, Dormand-Prince)
  + check namespace
  + warp-level collectives (study libcu++, MordenGPU for kernel)
* Data
  + add loader plug-in API - CIFAR
    - [Howto](https://franky07724-57962.medium.com/once-upon-a-time-in-cifar-10-c26bb056b4ce)
    - [Different Layers](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)
    - ImageDataGenerator (torchvision.transforms: resize, center-crop, shift, flip, width-change)
  + add K-fold sampler
* Model
  + GAN
    - [CIFAR-10](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/)
    - [Pytorch DCGAN](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    - [CelebA](https://medium.com/@manoharmanok/implementing-dcgan-in-pytorch-using-the-celeba-dataset-a-comprehensive-guide-660e6e8e29d2)
    - [AC-GAN](https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/)
    - use pre-trained model, i.e. [transfer learning](https://openaccess.thecvf.com/content_ECCV_2018/papers/yaxing_wang_Transferring_GANs_generating_ECCV_2018_paper.pdf)
    - torch.eval() i.e. normalize using running stat, disable dropout (vs torch.train())
  + New Layers
   - [Deconvolution](https://www.mdpi.com/2078-2489/15/11/711)
      * add [Transposed Convolution](https://d2l.ai/chapter_computer-vision/transposed-conv.html). Less used now b/c it creates checkerboard pattern, see https://distill.pub/2016/deconv-checkerboard/)
      * 1x1 Convolution (resize #channel)
   - residual net i.e. [ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html)
      * branch & concatenate (i.e Inception in GoogLeNet)
   - add Swish, Mish
* VM
  + CUDA 12 migration
    - Stream Management (cudaStreamAddCallback) and Event Management
    - EventSync/LaunchHostFunc, flip calling from GPU=>CPU (requires CUDA Stream + event pool)
    - dynamic Graph
    - CUB (now part of CCCL) again
  + TLSF using floating point [FP in Allocator](https://brnz.org/hbr/?p=1735)
  + Auto Differentiation i.e. JVP (forward), VJP (backward)
    - [autograd](https://github.com/HIPS/autograd)
    - [Jax](https://docs.jax.dev/en/latest/quickstart.html)
  + inter-VM communication (via CUDA stream)
  + inter-VM loader (from VM->VM)
  + free_tensor as linked-list (instead of an array)
* Design & Instrumentation
  + Llama
    x [llama2.c](https://www.signalpop.com/2024/02/10/understanding-llama2-c-and-chatgpt-a-visual-design-walkthrough/)
    - [Review](https://www.hostinger.com/tutorials/what-is-ollama). Local LLM environment with pre-train model.
    - [GGML Tensor library]( https://github.com/ggerganov/ggml). Host-oriented, review kernel code.
    - [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). Optimized for GPU, cross-platform, structured model storage.
* Model
  + Collections
    - Deep Layer Aggregration [DLA](https://arxiv.org/pdf/1707.06484)
  + Diffusion, [Stable Diffusion](https://stability.ai/). Pre-trained only?
  + Transformer
    - Review
      * [CNN vs ViT](https://arxiv.org/pdf/2406.03478) (good ref paper)
      * [seq2seq vs Transformer](https://towardsdatascience.com/neural-machine-translation-inner-workings-seq2seq-and-transformers-229faff5895b)
      * [Retentitive Network](https://arxiv.org/pdf/2307.08621)
    - Intro
      * [Attention is all you need](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
      * [lecture](https://courses.grainger.illinois.edu/ece448/sp2023/slides/lec24.pdf)
      * [what](https://www.datacamp.com/tutorial/how-transformers-work)
      * [cross-attention](https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78)
    - Code
      * [tensor core utilization, fully connectecd layer](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html)
      * [transformer](https://github.com/hyunwoongko/transformer?tab=readme-ov-file)
      * [position encoding](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
      * [python](https://benjaminwarner.dev/2023/07/01/attention-mechanism)
      * [pytorch](https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1)
      * [pytorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)
      * [llama.cpp](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file)
      * [llama2.c](https://github.com/karpathy/llama2.c/tree/b3c4b6c3c4bbff42e5211293280307019368ccb5?fbclid=IwY2xjawHhZS9leHRuA2FlbQIxMAABHcJp5Zx2VvEderi5aE7JRTtTrNiqe02gY-UOOveFiCvm_iMHgo8NRbj8QQ_aem__PtK6HblJyToUFr5Mov_dA). 700-line C. Tiny Llama trainning + inferencing.
  + RetNet
  + GNN - dynamic graph with VMs. Value proposition.
  + Mamba - State Space Model [mamba](https://www.ibm.com/think/topics/mamba-model)
  + Multi-Domain, i.e. MDNet

### LATER
* Tuning
  + Graph (CUDA 10.x) - host-only, to reduce repetitive launch overhead
  + HMM (CUDA 12.2) - unify CPU-GPU memory allocation
  + CUB (CUDA 12.4) - warp/block/device-wide collective primitives
* CUDA binary, study possibility for VM in assembly https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
  + Review [NVBits](https://github.com/NVlabs/NVBit?tab=readme-ov-file)
  + cuobjdump
  + cudisasm
* Language consistancy
  + compare to APL https://www.jsoftware.com/papers/APLDictionary.htm
  + compare to J, (rank & axis ops)
* Model
  + Latent Diffusion, [Stable Diffusion](https://stability.ai/). Pre-trained only?
  + RNN, **lost to Transformer**
* Inter-op
  + ONNX model exporter (protobuf), can be read by Netron
  + ONNX model importer, load pretrained models (from Model Zoo, Hugging Face)
* Data - **use ONNX instead**
  + NCHW tensor format support (as in PyTorch)
  + loader - .petastorm, .csv (available on github)
  + model persistence - .npy, .petastorm, hdf5
* Visualization - **use TensorBoard instead**
  + nvdiffrast https://nvlabs.github.io/nvdiffrast/
  + Netron - web-based NN model viewer (support .onnx, .pt, .h5)
  + OpenGL/WASM
* 3rd-party lib Integration
  + integrate CUB, CUTLASS (utilities.init, gemm_api) - slow, later
  + pre-processor (DALI) + GPUDirect - heavy, later
  + calling API - Python(cffi), Ruby(FFI)

## History
### [Release 1.0](./docs/v1_progress.md) features
* Dr. Ting's eForth words with F32 as data unit, U16 instruction unit
* Support parallel Forth VMs
* Lambda-based Forth microcode
* Memory management unit handles dictionary, stack, and parameter blocks in CUDA
* Managed memory debug utilities, words, see, ss_dump, mem_dump
* String handling utilities in CUDA
* Light-weight vector class, no dependency on STL
* Output Stream, async from GPU to host

### [Release 2.0](./docs/v2_progress.md) features
* vector, matrix, tensor objects (modeled to PyTorch)
* TLSF tensor storage manager (now 4G max)
* matrix arithmetic (i.e. +, -, *, copy, matmul, transpose)
* matrix fill (i.e. zeros, ones, fill, eye, random)
* matrix console input (i.e. matrix[..., array[..., and T![)
* matrix print (i.e PyTorch-style, adjustable edge elements)
* tensor view (i.e. dup, over, pick, r@)
* GEMM (i.e. a * A x B + b * C, use CUDA Dynamic Parallelism)
* command line option: debug print level control (T4_DEBUG)
* command line option: list (all) device properties
* use cuRAND kernel randomizer for uniform and standard normal distribution

### [Release 3.0](./docs/v3_progress.md) features
* NN model creation and persistence
* NN model batch control (feed forward, backprop w/ autograd)
* optimization - sgd
* layers - conv2d, linear, flatten
* pooling - maxpool, minpool, avgpool, dropout
* activation - relu, sigmoid, softmax, log_softmax
* loss - ce, mse, nll
* formatted data - NHWC (as in TensorFlow)
* dataset rewind
* mini-batch fetch
* dataset loader - MNIST
* OpenGL dataset Viewer

### [Release 3.2](./docs/v3_progress.md)
* NN model - supports GAN
* optimization - adam, sgd with momentum, grad_zero
* layers - conv1x1, upsample, batchnorm
* activation - tanh, selu, leakyrelu, elu
* loss - bce
* tensor op - std (stdvar), sqrt

