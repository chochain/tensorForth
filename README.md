## tensorForth - Forth does matrices and machine learning
* Forth VM that supports tensor calculus and Convolution Neural Network with dynamic parallelism in CUDA

### Status
|version|feature|stage|description|conceptual comparable|
|---|---|---|---|---|
|[release 1.0](https://github.com/chochain/tensorForth/releases/tag/v1.0.2)|**float**|beta|extended eForth with F32 float|Python|
|[release 2.0](https://github.com/chochain/tensorForth/releases/tag/v2.0.2)|**matrix**|alpha|added vector and matrix objects|NumPy|
|[release 2.2](https://github.com/chochain/tensorForth/releases/tag/v2.2.2)|**lapack**|alpha|added linear algebra methods|SciPy|
|[release 3.0](https://github.com/chochain/tensorForth/releases/tag/v3.0.0)|**CNN**|alpha|added ML propegation with autograd|Torch|
|next|**Transformer**|planning|add Transformer ops|PyTorch|

### Why?
Compiled programs run fast on Linux. On the other hand, command-line interface and shell scripting tie them together in operation. With interactive development, small tools are built along the way, productivity usually grows with time, especially in the hands of researchers.

*Niklaus Wirth*: **Algorithms + Data Structures = Programs**
* Too much on Algorithms - most modern languages, i.e. OOP, abstraction, template, ...
* Too focused on Data Structures - APL, SQL, ...

*Numpy* kind of solves both. So, for AI projects today, we use *Python* mostly. However, when GPU got involved, to enable processing on CUDA device, say with *Numba* or the likes, mostly there will be a behind the scene 'just-in-time' transcoding to C/C++ followed by compilation then load and run. In a sense, your *Python* code behaves like a *Makefile* which requires compilers/linker available on the host box. Common practice for code analysis can only happen at the tail-end after execution. This is usually a long journey. After many coffee breaks, we tweek the *Python* code and restart again. To monitor progress or catch any anomaly, scanning the intermittent dump become a habit which probably reminisce the line-printer days for seasoned developers. So much for 70 years of software engineering progress.

Forth language encourages incremental build and test. Having a 'shell', resides in GPU, that can interactively and incrementally develop/run each AI layer/node as a small 'subroutine' without dropping back to host system might better assist building a rapid and accurate system. The rationale is not unlike why the NASA probes sent into space are equipped with Forth chips. On the flipped side, with this kind of CUDA kernel code, some might argue that the branch divergence could kill the GPU. Well, the performance of the 'shell scripts' themselves are not really the point. So, here we are!

> **tensor + Forth = tensorForth!**

### How?
GPU, behaves like a co-processor or a DSP chip. It has no OS, no string support, and runs its own memory. Most of the available libraries are built for host instead of device i.e. to initiate calls from CPU into GPU but not the other way around. So, to be interactive, a memory manager, IO, and syncing with CPU are things needed to be had. It's pretty much like creating a Forth from scratch for a new processor as in the old days.

Since GPUs have good compiler support nowadays and I've ported the latest [*eForth*](https://github.com/chochain/eforth) to lambda-based in C++, pretty much all words can be transferred straight forward. However, having *FP32* or *float32* as my basic data unit, so later I can morph them to *FP16*, or even fixed-point, there are some small stuffs such as addressing and logic ops that require some attention.

The codebase will be in C for my own understanding of the multi-trip data flows. In the future, the class/methods implementation can come back to Forth in the form of loadable blocks so maintainability and extensibility can be utilized as other self-hosting systems. It would be amusing to find someone brave enough to work the assembly (i.e. CUDA SASS) into a Forth that resides on GPU micro-cores in the fashion of [*GreenArray*](https://www.greenarraychips.com/), or to forge an FPGA doing similar kind of things.

In the end, languages don't really matter. It's the problem they solve. Having an interactive Forth in GPU does not mean a lot by itself. However by adding vector, matrix, linear algebra support with a breath of **APL**'s massively parallel from GPUs. Neural Network tensor ops with backprop following the path from Numpy to PyTorch, plus the cleanness of **Forth**, it can be useful one day, hopefully! 

### Example - CNN Training on MNIST dataset
<pre>
10 28 28 1 nn.model                         \ create a network model (input dimensions)
0.5 10 conv2d 2 maxpool relu                \ add a convolution block
0.5 20 conv2d 0.5 dropout 2 maxpool relu    \ add another convolution block
flatten 0.0 49 linear                       \ add reduction layer, and
0.5 dropout 0.0 10 linear softmax           \ final fully connected output
constant md0                                \ we can store the model in a constant
                                
md0 batchsize dataset mnist_train           \ create a MNIST dataset with model batch size
constant ds0                                \ save dataset in a constant

variable acc 0 acc !                        \ create an accuracy counter, and zero it

\ here's the entire training framework in 3 lines
: cnn (N D -- N') for forward backprop nn.hit acc +! 0.01 0.0 nn.sgd 46 emit next ;
: stat cr . ." >" clock . ." : hit=" acc @ . 0 acc ! ." , loss=" loss.ce . cr ;
: epoch for cnn r@ stat ds0 rewind next ;

ds0                                         \ put dataset as TOS
19 epoch                                    \ execute multiple epoches
drop                                        \ drop dataset from TOS

nn.save tests/my_net.t4                     \ persist the trained network
</pre>

### Example - Small Matrix ops
<pre>
> ten4                # enter tensorForth
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], vmss[64*1], pmem=48K, tensor=1024M
2 3 matrix{ 1 2 3 4 5 6 }            \ create a 2x3 matrix
 <0 T2[2,3]> ok                      \ 2-D tensor shown on top of stack (TOS)
dup                                  \ duplicate
 <0 T2[2,3] T2[2,3]> ok              \ two matrices now sit on stack
.                                    \ print one
matrix[2,3] = {
	{ +1.0000 +2.0000 +3.0000 }
	{ +4.0000 +5.0000 +6.0000 } }
 <0 T2[2,3]> ok                      \ only one matrix now left on stack
3 2 matrix ones                      \ create a 3x2 matrix, fill it with ones
 <0 T2[2,3] T2[3,2]> ok
@                                    \ multiply matrices 2x3 @ 3x2
 <0 T2[2,3] T2[3,2] T2[2,2]> ok      \ 2x2 resultant matrix shown on TOS
.                                    \ print the matrix
matrix[2,2] = {
	{ +6.0000 +6.0000 }
	{ +15.0000 +15.0000 } }
 <0 T2[2,3] T2[3,2]> ok
2drop                                \ free both matrics
 <0> ok
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
2048 / .                             \ scale down and print the resultant 1024x512 matrix
matrix[1024,512] = {                 \ in PyTorch style (edgeitem=3)
	{ +0.4873 +0.4873 +0.4873 ... +0.4873 +0.4873 +0.4873 }
	{ +0.4274 +0.4274 +0.4274 ... +0.4274 +0.4274 +0.4274 }
	{ +0.5043 +0.5043 +0.5043 ... +0.5043 +0.5043 +0.5043 }
	...
	{ +0.5041 +0.5041 +0.5041 ... +0.5041 +0.5041 +0.5041 }
	{ +0.5007 +0.5007 +0.5007 ... +0.5007 +0.5007 +0.5007 }
	{ +0.5269 +0.5269 +0.5269 ... +0.5269 +0.5269 +0.5269 } }
 <0 T2[1024,2048] T2[2048,512] T2[1024,512] 2048> ok  \ note that matrix and 2048 are untouched
2drop                                       \ because tensor ops are by default non-destructive
 <0 T2[1024,2048] T2[2048,512]> ok          \ so we drop them from TOS

: mx clock >r for @ drop next clock r> - ;  \ now let's define a word 'mx' for benchmark loop
9 mx                                        \ run benchmark for 10 loops
 <0 T2[1024,2048] T2[2048,512] 396> ok      \ 396 ms for 10 cycles
drop                                        \ drop the value
 <0 T2[1024,2048] T2[2048,512]> ok
999 mx                                      \ now try 1000 loops
 <0 T2[1024,2048] T2[2048,512] 3.938+04> ok \ that is 39.38 sec (i.e. ~40ms / loop)
</pre>

### To build
* install CUDA 11.6 on your machine
* download one of the releases from the list above to your local directory

#### with Makefile, and test
* cd to your ten4 repo directory,
* update root Makefile to your desired CUDA_ARCH, CUDA_CODE,
* type 'make all',
* if all goes well, some warnings aside, cd to tests directory,
* enter the following for Forth ops<br/>
  > ~/tests> ten4 < lesson_1.txt - for basic syntax checks
* enter the following for testing matrix ops<br/>
  > ~/tests> ten4 < lesson_2.txt - for matrix ops,<br/>
  > ~/tests> ten4 < lesson_3.txt - for linear algebra stuffs
* enter the following for testsing machine learning ops<br/>
  > ~/tests> ten4 < lesson_4.txt - for single pass of forward, loss, and backprop<br/>
  > ~/tests> ten4 < lesson_5.txt - for full blown 20 epoches of MNIST training<br/>

#### with Eclipse
* install Eclipse
* install CUDA SDK 11.6 for Eclipse (from Nvidia site)
* create project by importing from your local repo root
* exclude directories - ~/tests, ~/img
* set File=>Properties=>C/C++ Build=>Setting=>NVCC compiler
  + Dialect=C++14
  + CUDA=5.2 or above
  + Optimization=O3

## tensorForth command line options
* \-h             - list all GPU id and their properties<br/>
* \-d device_id   - select GPU device id
* \-v verbo_level - set verbosity level 0: off (default), 1: mmu tracing on, 2: detailed trace

## Machine Learning volcabularies (see [doc3](./docs/v3_progress.md) for detail and examples)
### Model creation and persistence
<pre>
  nn.model   (n h w c -- N)      - create a Neural Network model with (n,h,w,c) input
  nn.load    (N -- N')           - load trained network from a given file name
  nn.save    (N -- N)            - export network as a file
  
  >n         (N T -- N')         - manually add tensor to model
  n@         (N n -- N T)        - fetch layered tensor from model, -1 is the latest layer
  network    (N -- N)            - display network model
</pre>
    
### Dataset and Batch controls
<pre>
  dataset    (n -- D)            - create a dataset with batch size = n, and given name i.e. 10 dataset abc
  fetch      (D -- D')           - fetch a mini-batch from dataset on return stack
  rewind     (D -- D')           - rewind dataset internal counters (for another epoch)
  batchsize  (D -- D b)          - get input batch size of a model
  
  forward    (N -- N')           - execute one forward path with rs[-1] dataset, layer-by-layer in given model
  forward    (N ds -- N')        - execute one forward propagation with TOS dataset, layer-by-layer in given model
  backprop   (N -- N')           - execute one backward propagation, adding derivatives for all parameters
  backprop   (N T -- N')         - execute one backward propagation with given onehot vector
  
  for        (N ds -- N')        - loop through a dataset, ds will be pushed onto return stack
  next       (N -- N')           - loop if any subset of dataset left, or ds is pop off return stack
</pre>

### CNN Layers (destructive by default)
<pre>
  conv2d     (N -- N')           - create a 2D convolution 3x3 filter, stride=1, padding=same, dilation=0, bias=0.5
  conv2d     (N b c -- N')       - create a 2D convolution, bias=b, c channels output, with default 3x3 filter
  conv2d     (N b c A -- N')     - create a 2D convolution, bias=b, c channels output, with config i.g. Vector[5, 5, 3, 2, 1] for (5x5, padding=3, stride=2, dilation=1, bais=0.3)
  flatten    (N -- N')           - flatten a tensor (usually input to linear)
  linear     (N b n -- N')       - linearize (y = Wx + b) from Ta input to n out_features
  
  maxpool    (N n -- N')         - nxn cells maximum pooling
  avgpool    (N n -- N')         - nxn cells average pooling
  minpool    (N n -- N')         - nxn cell minimum pooling
  dropout    (N p -- N')         - zero out p% of channel data (add noise between data points)
</pre>

### Activation and Loss (non-linear)
<pre>
  tanh       (Ta -- Ta')         - tensor element-wise tanh Ta' = tanh(Ta)
  relu       (Ta -- Ta')         - tensor element-wise ReLU Ta' = max(0, Ta)
  sigmoid    (Ta -- Ta')         - tensor element-wise Sigmoid Ta' = sigmoid(Ta)
  tanh       (N -- N')           - add tanh layer to network model
  relu       (N -- N')           - add Rectified Linear Unit to network model
  sigmoid    (N -- N')           - add sigmoid 1/(1+exp^-z) activation to network model, used in binary
  softmax    (N -- N')           - add probability vector exp(x)/sum(exp(x)) to network model, feeds loss.ce, used in multi-class
  logsoftmax (N -- N')           - add probability vector x - log(sum(exp(x))) to network model, feeds loss.nll, used in multi-class
</pre>

* Loss and Gradiant ops
<pre>
  loss.mse   (N Ta -- N Ta')     - mean squared error, take output from linear layer
  loss.ce    (N Ta -- N Ta')     - cross-entropy, takes output from softmax activation
  loss.nll   (N Ta -- N Ta')     - negative log likelihood, takes output from log-softmax activation
  
  nn.sgd     (N p m -- N')       - apply SGD(learn_rate=p, momentum=m) model back propagation
  nn.adam    (N a b1 -- N')      - apply Adam backprop alpha, beta1, default beta2=1-(1-b1)^3
  nn.adam    (N a b1 b2 -- N')   - apply Adam backprop with given alpha, beta1, beta2
  nn.onehot  (N -- N T)          - get cached onehot vector from a model
  nn.hit     (N -- N n)          - get number of hit (per mini-batch) of a model
</pre>

## Tensor Calculus volcabularies (see [doc2](./docs/v2_progress.md) for detail and examples)
### Tensor creation
<pre>
   vector    (n       -- T1)     - create a 1-D array and place on top of stack (TOS)
   matrix    (h w     -- T2)     - create 2-D matrix and place on TOS
   tensor    (n h w c -- T4)     - create a 4-D NHWC tensor on TOS
   vector{   (n       -- T1)     - create 1-D array from console stream
   matrix{   (h w     -- T2)     - create a 2-D matrix from console stream
   view      (Ta      -- Ta Va)  - create a view (shallow copy) of a tensor
   copy      (Ta      -- Ta Ta') - duplicate (deep copy) a tensor on TOS
</pre>

### Duplication ops (reference creation)
<pre>
   dup       (Ta    -- Ta Ta)    - create a reference of a tensor on TOS
   over      (Ta Tb -- Ta Tb Ta) - create a reference of the 2nd item (NOS)
   2dup      (Ta Tb -- Ta Tb Ta Tb)
   2over     (Ta Tb Tc Td -- Ta Tb Tc Td Ta Tb)
</pre>

### Tensor/View print
<pre>
   . (dot)   (Ta -- )        - print a vector, matrix, or tensor
   . (dot)   (Va -- )        - print a view of a tensor
</pre>

### Shape adjustment (change shape of origial tensor or view)
<pre>
   flatten   (Ta -- T1a')    - reshap a tensor or view to 1-D array
   reshape2  (Ta -- T2a')    - reshape to a 2-D matrix view
   reshape4  (Ta -- T4a')    - reshape to a 4-D NHWC tensor or view
</pre>

### Fill tensor with init values (data updated to original tensor)
<pre>
   zeros     (Ta   -- Ta')   - fill tensor with zeros
   ones      (Ta   -- Ta')   - fill tensor with ones
   full      (Ta   -- Ta')   - fill tensor with number on TOS
   eye       (Ta   -- Ta')   - fill diag with 1 and other with 0
   rand      (Ta   -- Ta')   - fill tensor with uniform random numbers
   randn     (Ta   -- Ta')   - fill tensor with normal distribution random numbers
   ={        (Ta   -- Ta')   - fill tensor with console input from the first element
   ={        (Ta n -- Ta')   - fill tensor with console input starting at n'th element
</pre>

### Tensor slice and dice
<pre>
   slice     (Ta i0 i1 j0 j1 -- Ta Ta') - numpy.slice[i0:i1,j0:j1,]
</pre>

### Tensor arithmetic (by default non-destructive)
<pre>
   +         (Ta Tb -- Ta Tb Tc)  - tensor element-wise addition Tc = Ta + Tb
   +         (Ta n  -- Ta n  Ta') - tensor-scalar addition (broadcast) Ta' = Ta + n
   +         (n  Ta -- n  Ta Ta') - scalar-tensor addition (broadcast) Ta' = Ta + n
   -         (Ta Tb -- Ta Tb Tc)  - tensor element-wise subtraction Tc = Ta - Tb
   -         (Ta n  -- Ta n  Ta') - tensor-scalar subtraction (broadcast) Ta' = Ta - n
   -         (n  Ta -- n  Ta Ta') - tensor-scalar subtraction (broadcast) Ta' = n - Ta
   @         (Ta Tb -- Ta Tb Tc)  - matrix-matrix inner product Tc = Ta @ Tb, i.e. matmul
   @         (Ta Ab -- Ta Ab Ac)  - matrix-vector inner product Ac = Ta @ Ab
   @         (Aa Ab -- Aa Ab n)   - vector-vector inner product n = Aa @ Ab, i.e. dot
   *         (Ta Tb -- Ta Tb Tc)  - tensor-tensor element-wise multiplication Tc = Ta * Tb
   *         (Ta Ab -- Ta Ab Ta') - matrix-vector multiplication Ta' = Ta * colum_vector(Ab)
   *         (Ta n  -- Ta n  Ta') - tensor-scalar multiplication Ta' = n * Ta, i.e. scale up
   *         (n  Ta -- n  Ta Ta') - scalar-tensor multiplication Ta' = n * Ta, i.e. scale up
   /         (Ta Tb -- Ta Tb Tc)  - tensor-tensor element-wise divide Tc = Ta / Tb
   /         (Ta n  -- Ta n  Ta') - tensor-scalar scale down Ta' = 1/n * Ta
   sum       (Ta    -- Ta n)      - sum all elements of a tensor
   avg       (Ta    -- Ta n)      - average of all elements of a tensor
   max       (Ta    -- Ta n)      - max of all elements of a tensor
   min       (Ta    -- Ta n)      - min of all elements of a tensor
</pre>

### Tensor arithmetic (by default destructive, as in Forth)
<pre>
   abs       (Ta    -- Ta')   - tensor element-wise absolute Ta' = abs(Ta)
   negate    (Ta    -- Ta')   - tensor element-wise negate   Ta' = -(Ta)
   exp       (Ta    -- Ta')   - tensor element-wise exponential Ta' = exp(Ta)
   +=        (Ta Tb -- Tc)    - tensor element-wise addition Tc = Ta + Tb
   +=        (Ta n  -- Ta')   - tensor-scalar addition (broadcast) Ta' = Ta + n
   +=        (n  Ta -- Ta')   - scalar-tensor addition (broadcast) Ta' = Ta + n
   -=        (Ta Tb -- Tc)    - tensor element-wise subtraction Tc = Ta - Tb
   -=        (Ta n  -- Ta')   - tensor-scalar subtraction (broadcast) Ta' = Ta - n
   -=        (n  Ta -- Ta')   - scalar-tensor subtraction (broadcast) Ta' = n - Ta
   @=        (Ta Tb -- Tc)    - matrix-matrix inner product Tc = Ta @ Tb, i.e. matmul
   @=        (Ta Ab -- Ac)    - matrix-vector inner product Ac = Ta @ Ab
   @=        (Aa Ab -- Ac)    - vector-vector inner prodcut n = Aa @ Ab, i.e. dot
   *=        (Ta Tb -- Tc)    - matrix-matrix element-wise multiplication Tc = Ta * Tb
   *=        (Ta Ab -- Ac')   - matrix-vector multiplication Ac' = Ta * Ab
   *=        (Ta n  -- Ta')   - tensor-scalar multiplication Ta' = n * Ta
   *=        (n  Ta -- Ta')   - scalar-tensor multiplication Ta' = n * Ta
   /=        (Ta Tb -- Tc)    - matrix-matrix element-wise Tc = Ta / Tb 
   /=        (Ta n  -- Ta')   - tensor-scalar scale down multiplication Ta' = 1/n * Ta
</pre>

### Linear Algebra (by default non-destructive)
<pre>
   matmul    (Ma Mb -- Ma Mb Mc) - matrix-matrix multiplication Mc = Ma @ Mb
   matdiv    (Ma Mb -- Ma Mb Mc) - matrix-matrix division Mc = Ma @ inverse(Mb)
   inverse   (Ma    -- Ma Ma')   - matrix inversion (Gauss-Jordan with Pivot)
   transpose (Ma    -- Ma Ma')   - matrix transpose
   det       (Ma    -- Ma d)     - matrix determinant (with PLU)
   lu        (Ma    -- Ma Ma')   - LU decomposition (no Pivot)
   luinv     (Ma    -- Ma Ma')   - inverse of an LU matrix
   upper     (Ma    -- Ma Ma')   - upper triangle
   lower     (Ma    -- Ma Ma')   - lower triangle with diag filled with 1s
   solve     (Ab Ma -- Ab Ma Ax) - solve linear equation AX = B
   gemm      (a b Ma Mb Mc -- a b Ma Mb Mc') - GEMM Mc' = a * Ma * Mb + b * Mc
</pre>


### TODO - by priorities
* data
  + add loader plug-in API - CIFAR
  + add K-fold sampler
* VM
  + inter-VM communication (CUDA stream, review CUB again)
  + inter-VM loader (from VM->VM)
* refactor
  + study JAX
    - JIT (XLA)
    - auto parallelization (pmap)
    - auto vectorization (vmap)
    - auto diff (grad), diffrax (RK4, Dormand-Prince)
  + add namespace
  + warp-level collectives (study libcu++, MordenGPU for kernel)
* model
  + add layer - Gelu, Sigmoid, Tanh, BatchNorm
  + add gradiant - Adam, AGC
  + add Transformer (BLOOM)
    - https://towardsdatascience.com/neural-machine-translation-inner-workings-seq2seq-and-transformers-229faff5895b
    - https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    - https://nlp.seas.harvard.edu/2018/04/03/attention.html
  + consider multi-domain (i.e. MDNet)
  + consider RNN
  + consider GNN - dynamic graph with VMs

### LATER
* data
  + NCHW tensor format support (as in PyTorch)
  + loader - .petastorm, .csv (available on github)
  + model persistance - .npy, .petastorm, hdf5
  + common models (VGG-19, ResNet (i.e. skip-connect), compare to Keras)
  + integrate ONNX
* plot
  + integrate plots (matplotlib, tensorboard)
* code
  + integrate CUB, CUTLASS (utilities.init, gemm_api) - slow, later
  + preprocessor (DALI) + GPUDirect - heavy, later

## History
### [Release 1.0](./docs/v1_progress.md) features
* Dr. Ting's eForth words with F32 as data unit, U16 instruction unit
* Support parallel Forth VMs
* Lambda-based Forth microcode
* Memory mangement unit handles dictionary, stack, and parameter blocks in CUDA
* Managed memory debug utilities, words, see, ss_dump, mem_dump
* String handling utilities in CUDA
* Light-weight vector class, no dependency on STL
* Output Stream, async from GPU to host

### [Release 2.0](./docs/v2_progress.md) features
* vector, matrix, tensor objects (modeled to PyTorch)
* TLSF tensor storage manager (now 4G max)
* matrix arithmetics (i.e. +, -, *, copy, matmul, transpose)
* matrix fill (i.e. zeros, ones, full, eye, random)
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
* NN model optimization - sgd
* layers - conv2d, linear, flatten
* pooling - maxpool, minpool, avgpool, dropout
* activation-  relu, sigmoid, softmax, log_softmax
* loss - ce, mse, nll
* formated data - NHWC (as in TensorFlow)
* dataset rewind
* mini-batch fetch
* dataset loader - MNIST
* OpenGL dataset Viewer
