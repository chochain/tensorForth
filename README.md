## tensorForth - eForth does tensor calculus, implemented in CUDA.
* Forth VM that supports tensor calculus and dynamic parallelism

### Status
|version|feature|stage|description|conceptual comparable|
|---|---|---|---|---|
|[release 1.0](https://github.com/chochain/tensorForth/releases/tag/v1.0.2)|**float**|beta|extended eForth with F32 float|Python|
|[release 2.0](https://github.com/chochain/tensorForth/releases/tag/v2.0.2)|**matrix**|alpha|added vector and matrix objects|NumPy|
|[release 2.2](https://github.com/chochain/tensorForth/releases/tag/v2.2.2)|**lapack**|alpha|added linear algebra methods|SciPy|
|next|**CNN**|planning|add tensor NN ops with autograd|PyTorch|
|-|**RNN**|later|-|-|
|-|**Adaptive**|long|-|-|

### Why?
Compiled programs run fast on Linux. On the other hand, command-line interface and shell scripting tie them together. Productivity grows with this model especially for researchers.

For AI development today, we use Python mostly. To enable processing on CUDA device, say with Numba or the likes, mostly there will be 'just-in-time' compilations behind the scene then load and run. In a sense, your Python code behaves like a Makefile which requires compilers to be on the host box. At the tailend, to analyze, visualization can then be have. This is usually a long journey. After many coffee breaks, we tweek the Python code and restart again. In order to catch progress, scanning the intermediate formatted files sometimes become necessary which probably reminisce the line-printer days for seasoned developers.

Having a 'shell' that can interactively and incrementally run 'compiled programs' from within GPU directly without dropping back to host system might be useful. Even though some might argue that the branch divergence in kernel could kill the GPU, but performance of the script itself is not really the point. So, here we are!

### How?
GPU, behaves like a co-processor. It has no OS, no string support, and runs its own memory. Most of the available libraries are built for host instead kernel mode i.e. to call from CPU instead of from within GPU. So, to be interactive, a memory manager, IO, and syncing with CPU are things to be had. It's pretty much like creating a Forth from scratch for a new processor as in the old days.

Since GPUs have good compiler support nowaday and I've ported the latest [eForth](https://github.com/chochain/eforth) to lambda-based in C++, pretty much all words can be straight copy except some attention to those are affected by CELL being float32 such as addressing, logic ops. i.e. BRANCH, 0=, MOD, XOR would not work as expected.

Having an interactive Forth in GPU does not mean a lot by itself. However, by adding matrix ops, linear algebra support, and tensor with backprop, sort of taking the path of Numpy to PyTorch, combining the cleanness of Forth with the massively parallel nature of GPUs can be useful one day, hopefully!

### Small Example
<pre>
> ten4                # enter tensorForth
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], vmss[64*1], pmem=48K, tensor=1024M
2 3 matrix{ 1 2 3 4 5 6 }            \ create a 2x3 matrix
 <0 T2[2,3]> ok                      \ 2-D tensor shown on top of stack (TOS)
dup                                  \ duplicate i.e. create a view
 <0 T2[2,3] V2[2,3]> ok              \ the view sits on TOS
.                                    \ print the view
matrix[2,3] = {
	{ +1.0000 +2.0000 +3.0000 }
	{ +4.0000 +5.0000 +6.0000 } }
 <0 T2[2,3]> ok
3 2 matrix ones                      \ create a 3x2 matrix, fill it with ones
 <0 T2[2,3] T2[3,2]> ok
@                                    \ multiply matrices [2,3] @ [3,2]
 <0 T2[2,3] T2[3,2] T2[2,2]> ok      \ [2,2] resultant matrix shown on TOS
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

### Larger Example - benchmark [1024,2048] x [2048,512] 1000 loops
<pre>
1024 2048 matrix rand                \ create a [1024,2048] matrix with uniform random values
 <0 T2[1024,2048]> ok                
2048 512 matrix ones                 \ create another [2048,512] matrix filled with 1s
 <0 T2[1024,2048] T2[2048,512]> ok
@                                    \ multiply them and resultant matrix on TOS
 <0 T2[1024,2048] T2[2048,512] T2[1024,512]> ok
2048 / .                             \ scale down and print the resutant [1024,512] matrix
matrix[1024,512] = {                 \ in PyTorch style (edgeitem=3)
	{ +0.4873 +0.4873 +0.4873 ... +0.4873 +0.4873 +0.4873 }
	{ +0.4274 +0.4274 +0.4274 ... +0.4274 +0.4274 +0.4274 }
	{ +0.5043 +0.5043 +0.5043 ... +0.5043 +0.5043 +0.5043 }
	...
	{ +0.5041 +0.5041 +0.5041 ... +0.5041 +0.5041 +0.5041 }
	{ +0.5007 +0.5007 +0.5007 ... +0.5007 +0.5007 +0.5007 }
	{ +0.5269 +0.5269 +0.5269 ... +0.5269 +0.5269 +0.5269 } }
 <0 T2[1024,2048] T2[2048,512] T2[1024,512> ok     \ original T2[1024,512] is still left on TOS
drop                                               \ because tensor ops are by default non-destructive
 <0 T2[1024,2048] T2[2048,512]> ok                 \ so we drop it from TOS
: mx clock >r for @ drop next clock r> - ;         \ define a word 'mx' for benchmark loop
9 mx                                               \ run benchmark for 10 loops
 <0 T2[1024,2048] T2[2048,512] 396> ok             \ 396 ms for 10 cycles
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
*      'ten4 < lesson_2.txt' for matrix ops,
*      'ten4 < lesson_3.txt' for linear algebra stuffs

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

## Forth Tensor operations (see [doc](./docs/v2_progress.md) for detail and examples)
### Tensor creation
<pre>
   vector    (n       -- T1)     - create a 1-D array and place on top of stack (TOS)
   matrix    (h w     -- T2)     - create 2-D matrix and place on TOS
   tensor    (n h w c -- T4)     - create a 4-D NHWC tensor on TOS
   vector{   (n       -- T1)     - create 1-D array from console stream
   matrix{   (h w     -- T2)     - create a 2-D matrix from console stream
   copy      (Ta      -- Ta Ta') - duplicate (deep copy) a tensor on TOS
</pre>

### View creation
<pre>
   dup       (Ta    -- Ta Va)    - create a view of a tensor on TOS
   over      (Ta Tb -- Ta Tb Va) - create a view from 2nd item on stack
   2dup      (Ta Tb -- Ta Tb Va Vb)
   2over     (Ta Tb Tc Td -- Ta Tb Tc Td Va Vb)
</pre>

### Tensor/View print
<pre>
   . (dot)   (Ta -- )        - print a vector, matrix, or tensor
   . (dot)   (Va -- )        - print a view (of vector, matrix, or tensor)
</pre>
### Shape adjustment (change shape of origial tensor)
<pre>
   flatten   (Ta -- T1a')    - reshap a tensor to 1-D array
   reshape2  (Ta -- T2a')    - reshape a 2-D matrix
   reshape4  (Ta -- T4a')    - reshape to a 4-D NHWC tensor
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

### Linear Algebra (by default non-destructive, except luinv)
<pre>
   matmul    (Ma Mb -- Ma Mb Mc) - matrix-matrix multiplication Mc = Ma @ Mb
   matdiv    (Ma Mb -- Ma Mb Mc) - matrix-matrix division Mc = Ma @ inverse(Mb)
   inverse   (Ma    -- Ma Ma')   - matrix inversion (Gauss-Jordan with Pivot)
   transpose (Ma    -- Ma Ma')   - matrix transpose
   det       (Ma    -- Ma d)     - matrix determinant (with PLU)
   lu        (Ma    -- Ma Ma')   - LU decomposition (no Pivot)
   luinv     (Ma    -- Ma')      - inverse of an LU matrix (stored in-place)
   upper     (Ma    -- Ma Ma')   - upper triangle
   lower     (Ma    -- Ma Ma')   - lower triangle with diag filled with 1s
   solve     (Ab Ma -- Ab Ma Ax) - solve linear equation AX = B
   gemm      (a b Ma Mb Mc -- a b Ma Mb Mc') - GEMM Mc' = a * Ma * Mb + b * Mc
</pre>

### TODO
* backprop and autograd
* add CNN
  + study torch.nn, CUB (for kernel)
  + conv: ~pushing_the_limits_for_2d_conv..., shuffle reduction
  + benchmark (MNIST, CIFAR, Kaggle...)
* models load/save - (VGG-19, ResNet (i.e. skip-connect), compare to Keras)
* sampling and distribution
* refactor - add namespace
* add RNN
* add inter-VM communication (CUDA stream, review CUB again)
* add batch loader (from VM->VM)
* .petastorm, .csv loader (available on github)
* add GNN - dynamic graph with VMs

### LATER
* integrate plots (matplotlib, tensorboard)
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

### [Release 2.2](./docs/v2_2_progress.md) features
