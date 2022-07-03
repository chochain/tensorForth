## tensorForth - eForth does tensor calculus, implemented in CUDA.
* Forth VM that supports tensor calculus and dynamic parallelism

### Status
* **float**   - [release 1.0](https://github.com/chochain/tensorForth/releases/tag/v1.0.2) beta
* **matrix**  - [release 2.0](https://github.com/chochain/tensorForth/releases/tag/v2.0.0) alpha
* **CNN**     - planning
* **RNN**     - later

### Why?
Compiled programs run fast on Linux. On the other hand, command-line interface and shell scripting tie them together. Productivity grows with this model especially for researchers.

For AI development today, we use Python mostly. To enable processing on CUDA device, say with Numba or the likes, mostly there will be 'just-in-time' compilations behind the scene then load and run. In a sense, the Python code behaves like a Makefile which requires compilers to be on the host box. At the tailend, to analyze, visualization can then be have. This is usually a long journey. After many coffee breaks, we update the Python and restart again. In order to catch progress, scanning the intermediate formatted files sometimes become necessary which probably reminisce the line-printer days for seasoned developers.

Having a 'shell' that can interactively and incrementally run 'compiled programs' from within GPU directly without dropping back to host system might be useful. Even though some might argue that the branch divergence in kernel could kill the GPU, but performance of the script itself is not the point. So, here we are!

### Small Example
<pre>
> ten4                               # enter tensorForth
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], pmem=48K, tensor=1024M
\  VM[0] dict=0x7f56fe000a00, mem=0x7f56fe004a00, vss=0x7f56fe010a00

2 3 matrix[ 1 2 3 4 5 6 ]            \ create matrix
mmu#tensor(2,3) => size=6            \ optional debug traces
 <0 T2[2,3]> ok                      \ 2-D tensor shown on top of stack (TOS)
dup                                  \ duplicate i.e. create a view
mmu#view 0x7efc18000078 => size=6
 <0 T2[2,3] V2[2,3]> ok              \ view shown on TOS
.                                    \ print the view
matrix[2,3] = [
	[+1.0000, +2.0000, +3.0000],
	[+4.0000, +5.0000, +6.0000]]
 <0 T2[2,3]> ok
mmu#free(T2) size=6                  \ view released after print
 <0 T2[2,3]> ok
3 2 matrix ones                      \ create a [3,2] matrix and fill with ones
mmu#tensor(3,2) => size=6
 <0 T2[2,3] T2[3,2]> ok
*                                    \ multiply matrices [2,3] x [3,x]
mmu#tensor(2,2) => size=4            \ a [2,x] resultant matrix created
 <0 T2[2,3] T2[3,2] T2[2,2]> ok      \ shown on TOS
.                                    \ print the matrix
matrix[2,2] = [
	[+6.0000, +6.0000],
	[+15.0000, +15.0000]]
 <0 T2[2,3] T2[3,2]> ok
mmu#free(T2) size=4                  \ matrix release after print
2drop                                \ free both matrics
mmu#free(T2) size=6
mmu#free(T2) size=6
 <0> ok
bye                                  \ exit tensorForth
 <0 T2[2,3] T2[3,2]> ok
tensorForth 2.0 done.
</pre>

### Larger Example - benchmark [1024,2048] x [2048,512] 1000 loops
<pre>
1024 2048 matrix rand                \ create a [1024,2048] matrix with uniform random values
 <0 T2[1024,2048]> ok                
2048 512 matrix ones                 \ create another [2048,512] matrix filled with 1s
 <0 T2[1024,2048] T2[2048,512]> ok
*                                    \ multiply them and resultant matrix on TOS
 <0 T2[1024,2048] T2[2048,512] T2[1024,512]> ok
2048 / .                             \ scale down and print the resutant [1024,512] matrix
matrix[1024,512] = [                 \ in PyTorch style (edgeitem=3)
	[+0.4873, +0.4873, +0.4873, ..., +0.4873, +0.4873, +0.4873],
	[+0.4274, +0.4274, +0.4274, ..., +0.4274, +0.4274, +0.4274],
	[+0.5043, +0.5043, +0.5043, ..., +0.5043, +0.5043, +0.5043],
	...,
	[+0.5041, +0.5041, +0.5041, ..., +0.5041, +0.5041, +0.5041],
	[+0.5007, +0.5007, +0.5007, ..., +0.5007, +0.5007, +0.5007],
	[+0.5269, +0.5269, +0.5269, ..., +0.5269, +0.5269, +0.5269]]
 <0 T2[1024,2048] T2[2048,512] T2[1024,512> ok     \ original T2[1024,512] is still left on TOS
drop                                               \ because tensor ops are by default non-destructive
 <0 T2[1024,2048] T2[2048,512]> ok                 \ so we drop it from TOS
: mx clock >r for * drop next clock r> - ;         \ define a word 'mx' for benchmark loop
5 mx                                               \ run benchmark for 6 loops
 <0 T2[1024,2048] T2[2048,512] 236> ok             \ 236 ms for 6 cycles
drop                                               \ drop the value
 <0 T2[1024,2048] T2[2048,512]> ok
999 mx                                             \ now try 1000 loops
 <0 T2[1024,2048] T2[2048,512] 3.938+04> ok        \ that is 39.38 sec (i.e. ~40ms / loop)
</pre>

Note:
* cuRAND uniform distribution averaged 0.5 is doing OK.
* 39.4 ms per 1Kx1K matmul on GTX 1660 with naive implementation. PyTorch average 0.850 ms which is 50x faster. Luckily, CUDA matmul tuning methods are well known. TODO!

### To build
* install CUDA 11.6 on your machine
* clone repo to your local directory

#### with Makefile, and test
* cd to your ten4 repo directory
* update root Makefile to your desired CUDA_ARCH, CUDA_CODE
* type 'make all'
* if all goes well, some warnings aside, cd to tests
* type 'ten4 < lesson_1.txt' for Forth syntax check,
* and  'ten4 < lesson_2.txt' for matrix stuffs

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
* \--h - list all GPU id and their properties<br/>
* \--d - select GPU device id

## Forth Tensor operations (see [doc](./docs/v2_progress.md) for detail and examples)
### Tensor creation words
<pre>
   array     (n -- T1)       - create a 1-D array and place on top of stack (TOS)
   matrix    (h w -- T2)     - create 2-D matrix and place on TOS
   tensor    (n h w c -- T4) - create a 4-D NHWC tensor on TOS
   array[    (n -- T1)       - create 1-D array from console stream
   matrix[   (h w -- T2)     - create a 2-D matrix from console stream
   copy      (Ta -- Ta Ta')  - duplicate (deep copy) a tensor on TOS
</pre>
### View creation words
<pre>
   dup       (Ta -- Ta Va)   - create a view of a tensor on TOS
   over      (Ta Tb -- Ta Tb Va)
   2dup      (Ta Tb -- Ta Tb Va Vb)
   2over     (Ta Tb Tc Td -- Ta Tb Tc Td Va Vb)
</pre>
### Tensor/View print word
<pre>
   . (dot)   (Ta -- )        - print array
</pre>
### Shape adjusting words (change shape of origial tensor)
<pre>
   flatten   (Ta -- T1a')    - reshap a tensor to 1-D array
   reshape2  (Ta -- T2a')    - reshape a 2-D matrix
   reshape4  (Ta -- T4a')    - reshape to a 4-D NHWC tensor
</pre>
### Fill tensor with init values (data updated to original tensor)
<pre>
   T![       (Ta -- Ta')     - fill tensor with console input
   zeros     (Ta -- Ta')     - fill tensor with zeros
   ones      (Ta -- Ta')     - fill tensor with ones
   full      (Ta -- Ta')     - fill tensor with number on TOS
   eye       (Ta -- Ta')     - fill diag with 1 and other with 0
   rand      (Ta -- Ta')     - fill tensor with uniform random numbers
   randn     (Ta -- Ta')     - fill tensor with normal distribution random numbers
</pre>
### Matrix arithmetic words (by default non-destructive)
<pre>
   +         (Ta Tb -- Ta Tb Tc) - tensor element-wise addition
   +         (Ta n  -- Ta Ta')   - tensor matrix-scaler addition (broadcast)
   -         (Ta Tb -- Ta Tb Tc) - tensor element-wise subtraction
   -         (Ta n  -- Ta Ta')   - tensor matrix-scaler subtraction (broadcast)
   *         (Ta Tb -- Ta Tb Tc) - matrix-matrix multiplication
   *         (Ta Ab -- Ta Ab Ta')- TODO: matrix-array multiplication (broadcase)
   *         (Aa Ab -- Aa Ab n)  - array-array dot product
   *         (Ta n  -- Ta Ta')   - matrix-scaler multiplication (broadcast)
   /         (Ta Tb -- Ta Tb Tc) - TODO: A * inv(B) matrix
   /         (Ta n  -- Ta Ta')   - matrix-scaler scale down multiplication (broadcast)
   inverse   (Ta    -- Ta Ta')   - TODO: matrix inversion
   transpose (Ta    -- Ta Ta')   - matrix transpose
   matmul    (Ta Tb -- Ta Tb Tc) - matrix multiplication
   gemm      (a b Ta Tb Tc -- a b Ta Tb Tc') - GEMM Tc' = a * Ta x Tb + b * Tc
</pre>

### TODO
* .npy loader/saver
* dataset iterator?
* tensor gradiant and backprop
* NN (study torch.nn, CUB (for kernel))
  + word as a net container (serves both sequential and functional)
  + CNN (2d)
    - conv: ~pushing_the_limits_for_2d_conv..., shuffle reduction
    - activation (relu, softmax): 
    - pooling (max): max of 2x2
    - linear (y=Wx+b): mm
    - dropout
  + loss       - nll (negative likelihood), mse (mean square error), ce (cross-entropy)
  + optimizer  - sgd (stochastic gradiant decent), adam
* ML cases and benchmark (MNIST, CIFAR, Kaggle...)
* sampling and distribution
* refactor - add namespace
* add inter-VM communication (CUDA stream)
* add GNN - dynamic graph with VMs

### LATER
* formatted file IO (.petastorm, .csv) - available on github but later
* integrate plots (tensorboard, R)
* integrate ONNX
* integrate CUB, CUTLASS (utilities.init, gemm_api) - slow, later
* preprocessor (DALI) + GPUDirect - heavy, later

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
* array, matrix, tensor objects (modeled to PyTorch)
* TLSF tensor storage manager (now 4G max)
* matrix arithmetics (i.e. +, -, *, copy, matmul, transpose)
* matrix fill (i.e. zeros, ones, full, eye, random)
* matrix console input (i.e. matrix[..., array[..., and T![)
* matrix print (i.e PyTorch-style, adjustable edge elements)
* tensor view (i.e. dup, over, pick, r@)
* GEMM (i.e. a * A x B + b * C, use CUDA Dynamic Parallelism)
* command line option: debug print level control (MMU_DEBUG)
* command line option: list (all) device properties
* use cuRAND kernel randomizer for uniform and standard normal distribution

