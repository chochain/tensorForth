# tensorForth - Release 2.0 / 2022-07
## Features
* array, matrix, tensor objects (modeled to PyTorch)
* TLSF tensor storage manager
* matrix arithmetics (i.e. +, -, *, copy, matmul, transpose)
* matrix fill (i.e. zeros, ones, full, eye, random)
* matrix console input (i.e. matrix[..., array[..., and T![)
* matrix print (i.e PyTorch-style, adjustable edge elements)
* tensor view instead of deep copy (i.e. dup, over, pick, r@, )
* GEMM (i.e. a * A x B + b * C, use CUDA Dynamic Parallelism)
* command line option: debug print level control (MMU_DEBUG)
* command line option: list (all) device properties
* use cuRAND kernel randomizer for uniform and standard normal distribution

## tensorForth Command line options
* \--h - list all GPU id and their properties<br/>
Example:> ./ten4 \--h<br/>
<pre>
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
* \--d - enter device id
Example:> ./ten4 \--d=0
<pre>
tensorForth 2.0
\  GPU 0 initialized at 1800MHz, dict[1024], pmem=48K, tensor=1024M
\  VM[0] dict=0x7fe3d2000a00, mem=0x7fe3d2004a00, vss=0x7fe3d2010a00
</pre>
* \--v - set verbosity level 0: off (default), 1: mmu tracing on, 2: detailed trace

## Forth Tensor operations
### Tensor creation ops
|word|param/example|tensor creation ops|
|---|---|---|
|array|(n -- T1)|create a 1-D array and place on top of stack (TOS)|
||> `5 `**`array`**|`T1[5]`|
|matrix|(h w -- T2)|create 2-D matrix and place on TOS|
||> `2 3`**`matrix`**|`T2[2,3]`|
|tensor|(n h w c -- T4)|create a 4-D NHWC tensor on TOS|
||> `64 224 224 3`**`tensor`**|`T4[64,224,224,3]`|
|array[|(n -- T1)|create 1-D array from console stream|
||> `5`**`array{`**`1 2 3 4 5 }`|`T1[5]`|
|matrix[|(h w -- T2)|create a 2-D matrix as TOS|
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
|. (dot)|(T1 -- )|print array|
||> `5 array{ 1 2 3 4 5 }`<br/>> **`.`**|`T1[5]`<br/>`array[5] = { +1.0000 +2.0000 +3.0000 +4.0000 +5.0000 }`|
|. (dot)|(T2 -- )|print matrix|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> **`.`**|`T2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`|
|. (dot)|(V2 -- )|print view|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `dup`<br/>> **`.`**|`T2[2,3]`<br/>`T2[2,3] V2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`|

### Shape adjusting ops
|word|param/example|Shape adjusting ops|
|---|---|---|
|flatten|(Ta -- Ta')|reshap a tensor to 1-D array|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> **`flatten`**<br/>> `.`|`T2[2,3]`</br>`T1[6]`<br/>`array[6] = { +1.0000 +2.0000 +3.0000 +4.0000 +5.0000 +6.0000 }`|
|reshape2|(h w Ta -- Ta')|reshape a 2-D matrix|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `dup .`<br/>> `3 2`**`reshape2`**</br>> `dup .`|`T2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`<br/>`T2[3,2]`<br/>`matrix[3,2] = { { +1.0000 +2.0000 } { +3.0000 +4.0000 } { +5.0000 +6.0000 } }`|
|reshape4|(n h w c Ta -- Ta')|reshape to a 4-D NHWC tensor|
||> `2 3 matrix{ 1 2 3 4 5 6 }`<br/>> `1 3 2 1`**`reshape4`**|`T2[2,3]`<br/>`T4[1,3,2,1]`|

### Fill ops
|word|param/example|Fill tensor with init valuess|
|---|---|---|
|zeros|(Ta -- Ta')|fill tensor with zeros|
||> `2 3 matrix`**`zeros`**<br/>> `.`|`T2[2,3]`<br/>`matrix[2,3] = { { +0.0000 +0.0000 +0.0000 } { +0.0000 +0.0000 +0.0000 } }`|
|ones|(Ta -- Ta')|fill tensor with ones|
||> `2 2 matrix`**`ones`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { +1.0000 +1.0000 } { +1.0000 +1.0000 } }`|
|full|(Ta n -- Ta')|fill tensor with number on TOS|
||> `2 2 matrix 3`**`full`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { +3.0000 +3.0000 } { +3.0000 +3.0000 } }`|
|eye|(Ta -- Ta')|TODO: fill diag with 1 and other with 0|
|rand|(Ta -- Ta')|fill tensor with uniform [0.00, 1.00) random numbers|
||> `2 2 matrix`**`rand`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { +0.5000 +0.1953 } { +0.1094 +0.4141 } }`|
|randn|(Ta -- Ta')|fill tensor with standard distribution random numbers|
||> `2 2 matrix`**`randn`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { -0.2170 -0.0695 } { -0.0107 +2.6791 } }`|
|={|(Ta -- Ta')|fill tensor with console input values|
||> `2 3 matrix`<br/>> **`={`**`1 2 3 4 5 6 }`<br/>> `.`|`T2[2,3]`<br/>`T2[2,3]`<br/>`matrix[2,3] = { { +1.0000 +2.0000 +3.0000 } { +4.0000 +5.0000 +6.0000 } }`|
|={|(Ta n -- Ta')|fill tensor from console starting at indexed item|
||> `2 3 matrix zeros`<br/>> **`2 ={`**`1 2 }`<br/>> `.`|`T2[2,3]`<br/>`T2[2,3]`<br/>`matrix[2,3] = { { +0.0000 +0.0000 +1.0000 } { +2.0000 +0.0000 +0.0000 } }`|

### Tensor slice and dice
|word|param/example|tensor slicing ops (non-destructive)|
|---|---|---|
|slice|(Ta x0 x1 y0 y1 -- Ta Ta')|numpy.slice[x0:x1, y0:y1, ]|
||> `4 4 matrix rand`<br/>> `dup .`<br/>> **`1 3 1 3 slice`**<br/>> `.`|`T2[4,4]`<br/>`matrix[4,4] = {`<br/> `{ +0.0940 +0.5663 +0.3323 +0.0840 }`<br/> `{ +0.6334 +0.3548 +0.1104 +0.7236 }`<br/> `{ +0.2781 +0.0530 +0.7532 +0.4145 }`<br/> `{ +0.4473 +0.0823 +0.1551 +0.3159 } }`<br> `matrix[2,2] = {`<br/> `{ +0.3548 +0.1104 }`</br> `{ +0.0530 +0.7532 } }`|

### Tensor Arithmetic ops
|word|param/example|Matrix arithmetic ops (non-destructive)|
|---|---|---|
|+|(Ta Tb -- Ta Tb Tc)|tensor element-wise addition|
||> `2 2 matrix random`<br/>> `dup .`<br/>> `2 2 matrix ones`<br/>> **`+`**<br/>> `.`|`T2[2,2]`<br/>`matrix[2,2] = { { -0.5000 +0.1953 } { +0.1094 +0.4141 } }`<br/>`T2[2,2] T2[2,2]`<br/>`T2[2,2] T2[2,2] T[2,2]`<br/>`matrix[2,2] = { { +0.5000 +1.1953 } { +1.1094 +1.4141 } }`|
|-|(Ta Tb -- Ta Tb Tc)|tensor element-wise subtraction|
|*|(Ta Tb -- Ta Tb Tc)|matrix-matrix multiplication|
|*|(Ta Ab -- Ta Ab Tc)|TODO: matrix-array multiplication|
|*|(Aa Ab -- Aa Ab c)|array-array dot product|
|*|(Ta v  -- Ta Ta')|matrix-scaler multiplication|
|/|(Ta Tb -- Ta Tb Tc)|TODO: C = A x inverse(B)|
|/|(Ta v  -- Ta Ta')|matrix-scaler division|
|sum|(Ta -- Ta n)|sum all elements of a tensor|
|exp|(Ta -- Ta Ta')|exponential (i.e. e^x) all elements of a tensor|
|inverse|(Ta -- Ta Ta')|TODO: matrix inversion|
|transpose|(Ta -- Ta Ta')|matrix transpose|
|matmul|(Ta Tb -- Ta Tb Tc)|matrix multiplication|
|gemm|(a b Ta Tb Tc -- a b Ta Tb Tc')|GEMM Tc' = a * Ta x Tb + b * Tc|

