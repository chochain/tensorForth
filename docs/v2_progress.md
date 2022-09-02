# tensorForth - Release 2.2 / 2022-08
## Features
* vector, matrix, tensor objects (modeled to PyTorch)
* TLSF tensor storage manager
* matrix arithmetics (i.e. +, -, @, *, sum, min, max, avg, abs, negate, exp, log, pow)
* linear algebra (i.e. copy, matmul, inverse, transpose, det, lu, luinv, upper, lower, solve)
* matrix fill (i.e. zeros, ones, full, eye, random)
* matrix console input (i.e. matrix{..., vector{..., and T!{)
* matrix print (i.e PyTorch-style, adjustable edge elements)
* tensor view instead of deep copy (i.e. dup, over, pick, r@)
* GEMM (i.e. a * A * B + b * C, use CUDA Dynamic Parallelism)
* command line option: debug print level control (MMU_DEBUG)
* command line option: list (all) device properties
* use cuRAND kernel randomizer for uniform and standard normal distribution

## tensorForth Command line options
* \-h - print usage and list all GPU id and their properties<br/>
* > Example:> ./ten4 \-h<br/>
<pre>
tensorForth - Forth does tensors, in GPU
Options:
  -h      list all GPUs and this usage statement.
  -d n    process using given device/GPU id
  -v n    Verbosity level, 0: default, 1: mmu debug, 2: more details

Examples:
$ ./tests/ten4 -h     ;# display help
$ ./tests/ten4 -d 0   ;# use device 0
$ ./tests/ten4 -v 1   ;# set verbosity to level 1

CUDA Device #0
	Name:                          NVIDIA GeForce GTX 1660
	CUDA version:                  7.5
	Total global memory:           5939M
	Total shared memory per block: 48K
	Number of multiprocessors:     22
	Total registers per block:     64K
	Warp size:                     32
	Max memory pitch:              2048M
	Max threads per block:         1024
	Max dim of block:              [1024, 1024, 64]
	Max dim of grid:               [2048M, 64K, 64K]
	Clock rate:                    1800KHz
	Total constant memory:         64K
	Texture alignment:             512
	Concurrent copy and execution: Yes
	Kernel execution timeout:      Yes
</pre>
* \-d device_id - enter GPU/device id
* > Example:> ./ten4 \-d 0
<pre>
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], vmss[64*1], pmem=48K, tensor=1024M
\  VM[0] dict=0x7fe3d2000a00, mem=0x7fe3d2004a00, vmss=0x7fe3d2010a00, obj=0x7fe460000000
</pre>
* \-v verbose_level - set verbosity level 0: off (default), 1: mmu tracing on, 2: detailed trace

## Example with tracing on
<pre>
> ten4 -v 1                          # enter tensorForth, with mmu debug tracing on
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], pmem=48K, tensor=1024M
\  VM[0] dict=0x7f56fe000a00, mem=0x7f56fe004a00, vss=0x7f56fe010a00

2 3 matrix{ 1 2 3 4 5 6 }            \ create matrix
mmu#tensor(2,3) => numel=6           \ the optional debug traces
 <0 T2[2,3]> ok                      \ 2-D tensor shown on top of stack (TOS)
dup                                  \ duplicate i.e. create a view
mmu#view => V2 numel=6
 <0 T2[2,3] V2[2,3]> ok              \ view shown on TOS
.                                    \ print the view
matrix[2,3] = {
	{ +1.0000 +2.0000 +3.0000 }
	{ +4.0000 +5.0000 +6.0000 } }
 <0 T2[2,3]> ok
mmu#free(T2) numel=6                 \ view released after print
 <0 T2[2,3]> ok
3 2 matrix ones                      \ create a [3,2] matrix and fill with ones
mmu#tensor(3,2) => numel=6
 <0 T2[2,3] T2[3,2]> ok
@                                    \ multiply matrices [2,3] x [3,x]
mmu#tensor(2,2) => numel=4           \ a [2,x] resultant matrix created
 <0 T2[2,3] T2[3,2] T2[2,2]> ok      \ shown on TOS
.                                    \ print the matrix
matrix[2,2] = {
	{ +6.0000 +6.0000 }
	{ +15.0000 +15.0000 } }
 <0 T2[2,3] T2[3,2]> ok
mmu#free(T2) numel=4                 \ matrix release after print
2drop                                \ free both matrics
mmu#free(T2) numel=6
mmu#free(T2) numel=6
 <0> ok
bye                                  \ exit tensorForth
 <0 T2[2,3] T2[3,2]> ok
tensorForth 2.0 done.
</pre>

## Forth Tensor operations
### Tensor creation ops
|word|param/example|tensor creation ops|
|---|---|---|
|vector|(n -- T1)|create a 1-D array and place on top of stack (TOS)|
||> `5 `**`vector`**|`T1[5]`|
|matrix|(h w -- T2)|create 2-D matrix and place on TOS|
||> `2 3`**`matrix`**|`T2[2,3]`|
|tensor|(n h w c -- T4)|create a 4-D NHWC tensor on TOS|
||> `64 224 224 3`**`tensor`**|`T4[64,224,224,3]`|
|vector{|(n -- T1)|create 1-D array from console stream|
||> `5`**`vector{`**`1 2 3 4 5 }`|`T1[5]`|
|matrix{|(h w -- T2)|create a 2-D matrix as TOS|
||> `2 3`**`matrix{`**`1 2 3 4 5 6 }`<br/>> `3 2`**`matrix{`**`{ 1 2 } { 3 4 } { 5 6 } }`|`T2[2,3]`</br>`T2[2,3] T2[3,2]`|
|copy|(Ta -- Ta Ta')|duplicate (deep copy) a tensor on TOS|
||> `2 3 matrix`<br/>> **`copy`**|`T2[2,3]`<br/>`T2[2,3] T2[2,3]`|

### Views creation ops
|word|param/example|view creation ops|
|---|---|---|
|dup|(Ta -- Ta Ta')|create a view of a tensor on TOS|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> **`dup`**|`T2[2,3]`<br/>`T2[2,3] V2[2,3]`|
|over|(Ta Tb -- Ta Tb Ta')||
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `3 2 matrix`<br/>> **`over`**|`T2[2,3]`<br/>`T2[2,3] T2[3,2]`<br/>`T2[2,3] T2[3,2] V2[2,3]`|
|2dup|(Ta Tb -- Ta Tb Ta' Tb')||
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `3 2 matrix`<br/>> **`2dup`**|`T2[2,3]`<br/>`T2[2,3] T2[3,2]`<br/>`T2[2,3] T2[3,2] V2[2,3] V2[3,2]`|
|2over|(Ta Tb Tc Td -- Ta Tb Tc Td Ta' Tb')|`...`|

### Tensor/View print
|word|param/example|Tensor/View print|
|---|---|---|
|. (dot)|(T1 -- )|print vector|
||> `5 vector{ 1 2 3 4 5 }`<br/>> **`.`**|`T1[5]`<br/>`vector[5] = { +1.0000 +2.0000 +3.0000 +4.0000 +5.0000 }`|
|. (dot)|(T2 -- )|print matrix|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> **`.`**|`T2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`|
|. (dot)|(V2 -- )|print view|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `dup`<br/>> **`.`**|`T2[2,3]`<br/>`T2[2,3] V2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`|

### Shape adjustment ops
|word|param/example|Shape adjusting ops|
|---|---|---|
|flatten|(Ta -- Ta')|reshap a tensor to 1-D array|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> **`flatten`**<br/>> `.`|`T2[2,3]`</br>`T1[6]`<br/>`vector[6] = { +1.0000 +2.0000 +3.0000 +4.0000 +5.0000 +6.0000 }`|
|reshape2|(h w Ta -- Ta')|reshape a 2-D matrix|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `dup .`<br/>> `3 2`**`reshape2`**</br>> `dup .`|`T2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`<br/>`T2[3,2]`<br/>`matrix[3,2] = { { +1.0000 +2.0000 } { +3.0000 +4.0000 } { +5.0000 +6.0000 } }`|
|reshape4|(n h w c Ta -- Ta')|reshape to a 4-D NHWC tensor|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `1 3 2 1`**`reshape4`**|`T2[2,3]`<br/>`T4[1,3,2,1]`|

### Tensor Fill ops
|word|param/example|Fill tensor with init valuess|
|---|---|---|
|zeros|(Ta -- Ta')|fill tensor with zeros|
||> `2 3 matrix`**`zeros`**<br/>> `.`|`T2[2,3]`<br/>`matrix[2,3] = { { +0.0000 +0.0000 +0.0000 } { +0.0000 +0.0000 +0.0000 } }`|
|ones|(Ta -- Ta')|fill tensor with ones|
||> `2 2 matrix`**`ones`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { +1.0000 +1.0000 } { +1.0000 +1.0000 } }`|
|full|(Ta n -- Ta')|fill tensor with number on TOS|
||> `2 2 matrix 3`**`full`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { +3.0000 +3.0000 } { +3.0000 +3.0000 } }`|
|eye|(Ta -- Ta')|fill diag with 1 and other with 0|
||> `3 3 matrix`**`eye`**<br/>> `.`|`T2[3,3]`<br/>`matrix[3,3] = {`<br/>`{ +1.0000 +0.0000 +0.0000 }`<br/>`{ +0.0000 +1.0000 +0.0000 }`<br/>`{ +0.0000 +0.0000 +1.0000 } }`|
|rand|(Ta -- Ta')|fill tensor with uniform [0.00, 1.00) random numbers|
||> `2 2 matrix`**`rand`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { +0.5000 +0.1953 } { +0.1094 +0.4141 } }`|
|randn|(Ta -- Ta')|fill tensor with standard distribution random numbers|
||> `2 2 matrix`**`randn`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { -0.2170 -0.0695 } { -0.0107 +2.6791 } }`|
|={|(Ta -- Ta')|fill tensor with console input values from the first element|
||> `2 3 matrix`<br/>> **`={`**`1 2 3 4 5 6 }`<br/>> `.`|`T2[2,3]`<br/>`T2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`|
|={|(Ta n -- Ta')|fill tensor from console starting at the indexed element|
||> `2 3 matrix zeros`<br/>> **`2 ={`**`1 2 }`<br/>> `.`|`T2[2,3]`<br/>`T2[2,3]`<br/>`matrix[2,3] = { { +0.0000 +0.0000 +1.0000 } { +2.0000 +0.0000 +0.0000 } }`|

### Tensor slice and dice
|word|param/example|tensor slicing ops (non-destructive)|
|---|---|---|
|slice|(Ta x0 x1 y0 y1 -- Ta Ta')|numpy.slice[x0:x1, y0:y1, ]|
||> `4 4 matrix rand`<br/>> `dup .`<br/>> **`1 3 1 3 slice`**<br/>> `.`|`T2[4,4]`<br/>`matrix[4,4] = {`<br/> `{ +0.0940 +0.5663 +0.3323 +0.0840 }`<br/> `{ +0.6334 +0.3548 +0.1104 +0.7236 }`<br/> `{ +0.2781 +0.0530 +0.7532 +0.4145 }`<br/> `{ +0.4473 +0.0823 +0.1551 +0.3159 } }`<br> `matrix[2,2] = {`<br/> `{ +0.3548 +0.1104 }`</br> `{ +0.0530 +0.7532 } }`|

### Tensor Arithmetic ops
|word|param/example|Tensor arithmetic ops (non-destructive)|
|---|---|---|
|+|(Ta Tb -- Ta Tb Tc)|tensor element-wise addition Tc = Ta + Tb|
||> `2 2 matrix rand`<br/>> `dup .`<br/>> `2 2 matrix ones`<br/>> **`+`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { -0.5000 +0.1953 } { +0.1094 +0.4141 } }`<br/>`T2[2,2] T2[2,2]`<br/>`T2[2,2] T2[2,2] T[2,2]`<br/>`matrix[2,2] = { { +0.5000 +1.1953 } { +1.1094 +1.4141 } }`|
|+|(Ta n  -- Ta n  Ta')|tensor-scalar addition (broadcast) Ta' = Ta + n|
|+|(n  Ta -- n  Ta Ta')|scalar-tensor addition (broadcast) Ta' = Ta + n|
|-|(Ta Tb -- Ta Tb Tc)|tensor element-wise subtraction Tc = Ta - Tb|
|-|(Ta n  -- Ta n  Ta')|tensor-scalar subtraction (boardcast) Ta' = Ta - n|
|-|(n  Ta -- n  Ta Ta')|scalar-tensor subtraction (boardcast) Ta' = n - Ta|
|@|(Ta Tb -- Ta Tb Tc)|matrix-matrix inner product Tc = Ta @ Tb|
|@|(Ta Ab -- Ta Tb Ac)|matrix-vector inner product Ac = Ta @ Ab|
|@|(Aa Ab -- Aa Ab n)|vector-vector inner product n = Aa @ Ab, i.e. dot|
|*|(Ta Tb -- Ta Tb Tc)|matrix-matrix element-wise multiplication Tc = Ta * Tb|
|*|(Ta n  -- Ta n Ta')|tensor-scalar multiplication Ta' = n * Ta, i.e. scale up|
|*|(n  Ta -- n  Ta Ta')|scalar-tensor multiplication Ta' = n * Ta, i.e. scale up|
|*|(Aa Ab -- Aa Ab Ac)|vector-vector outer product Ac = Aa * Ab|
|/|(Ta Tb -- Ta Tb Tc)|matrix-matrix element-wise division Tc = Ta / Tb|
|/|(Ta n  -- Ta Ta')|tensor-scalar division Ta = 1/n * Ta, i.e. scale down|
|sum|(Ta -- Ta n)|sum all elements of a tensor|
|avg|(Ta -- Ta n)|average all elements of a tensor|
|max|(Ta -- Ta n)|max of all elements of a tensor|
|min|(Ta -- Ta n)|min of all elements of a tensor|

### Tensor Arithmetic ops (self-assign, i.e. destructive as in Forth)
|word|param/example|Tensor arithmetic ops (destructive)|
|---|---|---|
|abs|(Ta -- Ta')|tensor element-wise absolute Ta' = abs(Ta)|
|negate|(Ta -- Ta')|tensor element-wise negate Ta' = -(Ta)|
|exp|(Ta -- Ta')|tensor element-wise exponential Ta' = exp(Ta)|
|log|(Ta -- Ta')|tensor element-wise natural logarithm Ta' = ln(Ta)|
|pow|(Ta n -- Ta')|tensor element-wise power Ta' = e^n(Ta)|
|+=|(Ta Tb -- Tc)|tensor element-wise addition Tc = Ta + Tb|
||> `2 2 matrix rand`<br/>> `dup .`<br/>> `2 2 matrix ones`<br/>> **`+=`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { -0.5000 +0.1953 } { +0.1094 +0.4141 } }`<br/>`T2[2,2] T2[2,2]`<br/>`T2[2,2]`<br/>`matrix[2,2] = { { +0.5000 +1.1953 } { +1.1094 +1.4141 } }`|
|+=|(Ta n  -- Ta')|tensor-scalar addition (broadcast) Ta' = Ta + n|
|+=|(n  Ta -- Ta')|scalar-tensor addition (broadcast) Ta' = Ta + n|
|-=|(Ta Tb -- Tc)|tensor element-wise subtraction Tc = Ta - Tb|
|-=|(Ta n  -- Ta')|tensor-scalar subtraction (boardcast) Ta' = Ta - n|
|-=|(n  Ta -- Ta')|scalar-tensor subtraction (boardcast) Ta' = n - Ta|
|@=|(Ta Tb -- Tc)|matrix-matrix inner product Tc = Ta @ Tb|
|@=|(Ta Ab -- Ac)|matrix-vector inner product Ac = Ta @ Ab|
|@=|(Aa Ab -- n)|vector-vector inner (dot) product n = Aa * Ab|
|*=|(Ta Tb -- Tc)|matrix-matrix element-wise multiplication Tc = Ta * Tb|
|*=|(Ta n  -- Ta')|tensor-scalar multiplication Ta' = n * Ta|
|*=|(n  Ta -- Ta')|scalar-tensor multiplication Ta' = n * Ta|
|*=|(Aa Ab -- Ac')|vector-vector multiplication Ac = Aa * Ab|
|/=|(Ta Tb -- Tc)|matrix-matrix element-wise division Tc = Ta / Tb|
|/=|(Ta n  -- Ta')|tensor-scalar division Ta = 1/n * Ta|

### Linear Algebra ops
|word|param/example|Matrix arithmetic ops (non-destructive)|
|---|---|---|
|matmul|(Ma Mb -- Ma Mb Mc)|matrix multiplication Mc = Ma @ Mb|
|matdiv|(Ma Mb -- Ma Mb Mc)|matrix division Mc = Ma @ inverse(Mb)|
||> `3 3 matrix{ 2 2 5 1 1 1 4 6 8 } copy`<br/>> **`matdiv`** `.`|`T2[3,3] T[3,3]`<br/>`matrix[3,3] = { { 1.0000 +0.0000 +0.0000 } { -0.0000 +1.0000 +0.0000 } { +0.0000 +0.0000 +1.0000 } }`|
|inverse|(Ma -- Ma Ma')|matrix inversion (Gauss-Jordan with Pivot)|
||> `3 3 matrix{ 2 2 5 1 1 1 4 6 8 }`<br/>> **`inverse`**<br/>> `.`|`T2[3,3]`<br/>`T2[3,3] T[3,3]`<br/>`matrix[3,3] = { { 0.3333 +2.3333 -0.5000 } { -0.6667 -0.6667 +0.5000 } { +0.3333 -0.6667 +0.0000 } }`|
|transpose|(Ma -- Ma Ma')|matrix transpose|
|det|(Ma -- Ma d)|matrix determinant (with PLU)|
||> `3 3 matrix{ 1 2 4 3 8 14 2 6 13 }`<br/>> **`det`**|`T2[3,3]`<br/>`T2[3,3] 6`|
|lu|(Ma -- Ma Ma')|LU decomposition, no Pivot|
||> `3 3 matrix{ 1 2 4 3 8 14 2 6 13 }`<br/>> **`lu`**`.`|`T2[3,3]`<br/>`matrix[3,3] = {`<br/>`{ +1.0000 +2.0000 +4.0000 }`<br/>`{ +3.0000 +2.0000 +2.0000 }`<br/>`{ +2.0000 +1.0000 +3.0000 } }`|
|luinv|(Ma -- Ma Ma')|inverse of an LU matrix (i.e. forward & backward)|
||> `3 3 matrix{ 1 2 4 3 8 14 2 6 13 }`<br/>> **`luinv`**`.`|`T2[3,3]`<br/>`matrix[3,3] = {`<br/>`{ +1.0000 -1.0000 -0.6667 }`<br/>`{ -3.0000 +0.5000 -0.3333 }`<br/>`{ +1.0000 -1.0000 +0.3333 } }`|
|upper|(Ma -- Ma Ma')|upper triangle|
||> `3 3 matrix{ 1 -1 -2 -3 5 -4 1 -1 4 }`<br>> **`upper`**`.`|`T2[3,3]`<br/>`matrix[3,3] = {`<br/>`{ +1.0000 -1.0000 -2.0000 }`<br/>`{ +0.0000 +5.0000 -4.0000 }`<br/>`{ +0.0000 +0.0000 +4.0000 } }`|
|lower|(Ma -- Ma Ma')|lower triangle with diag filled with 1s|
||> `3 3 matrix{ 1 -1 -2 -3 5 -4 1 -1 4 }`<br>> **`lower`**`.`|`T2[3,3]`<br/>`matrix[3,3] = {`<br/>`{ +1.0000 +0.0000 +0.0000 }`<br/>`{ -3.0000 +1.0000 +0.0000 }`<br/>`{ +1.0000 -1.0000 +1.0000 } }`|
|solve|(Ab Ma -- Ab Ma Ax)|solve linear equation AX = B|
||> `3 vector{ 1 1 1 }`<br>> `3 3 matrix{ 5 7 4 3 -1 3 6 7 5 }`<br>> **`solve`**<br>> `dup .`|`T1[3]`<br/>`T1[3] T2[3,3]`<br/>`T1[3] T2[3,3] T1[3]`<br/>`vector[3] = { +8.0000 -1.0000 -8.0000 }`|
|gemm|(a b Ma Mb Mc -- a b Ma Mb Mc')|GEMM Mc' = a * Ma * Mb + b * Mc|


