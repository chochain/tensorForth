## tensorForth - eForth does tensor calculus, implemented in CUDA.
* Forth VM that supports tensor calculus and dynamic parallelism

### Status
* float  - Alpha
* tensor - planning
* gemm   - todo

### Why?
Compiled programs run fast on Linux. On the other hand, command-line interface and shell scripting tie them together. Productivity grows with this model especially for researchers.

For AI development today, we use Python mostly. To enable processing on CUDA device, say with Numba or the likes, mostly there will be 'just-in-time' compilations behind the scene then load and run. In a sense, the Python code behaves like a Makefile which requires compilers to be on the host box. At the tailend, to analyze, visualization can then be have. This is usually a long journey. After many coffee breaks, we update the Python and restart again. In order to catch progress, scanning the intermediate formatted files sometimes become necessary which probably reminisce the line-printer days for seasoned developers.

Having a 'shell' that can interactively and incrementally run 'compiled programs' from within GPU directly without dropping back to host system might be useful. Even though some might argue that the branch divergence could kill, but performance of the script itself is not the point. So, here we are!

### To build
* install CUDA 11.6 on your machine
* clone repo to your local directory

#### with Makefile, and test
* cd to your ten4 repo directory
* update root Makefile to your desired CUDA_ARCH, CUDA_CODE
* type 'make all'
* if all goes well, some warnings aside, cd to tests
* type 'ten4 < lesson_1.txt'

#### with Eclipse
* install Eclipse
* install CUDA SDK 11.6 for Eclipse (from Nvidia site)
* create project by importing from your local repo root
* exclude directories - ~/tests, ~/img
* set File=>Properties=>C/C++ Build=>Setting=>NVCC compiler
  + Dialect=C++14
  + CUDA=5.2 or above
  + Optimization=O3

### TODO
* blas examples
* use cuRAND
* formatted file IO (CSV, Numpy)
* preprocessor (DALI)
* NN (torch.nn)
* ML cases and benchmark (kaggle.MNIST, ...)
* add inter-VM communication (CUDA stream)
* add dynamic graph (GNN)
* integrate plots (tensorboard, R)
* integrate ONNX 
* integrate CUB, CUTLASS (utilities.init, gemm_api) - checked but slow, use straight CDP

### History
#### [Release 1.0](./docs/v1_progress.md) features
* Dr. Ting's eForth words with F32 as data unit, U16 instruction unit
* Support parallel Forth VMs
* Lambda-based Forth microcode
* Memory mangement unit handles dictionary, stack, and parameter blocks in CUDA
* Managed memory debug utilities, words, see, ss_dump, mem_dump
* String handling utilities in CUDA
* Light-weight vector class, no dependency on STL
* Output Stream, async from GPU to host
#### Release 2.0 features
* array, matrix objects (modeled to PyTorch)
* TLSF tensor storage manager
* matrix addition, multiplication
* GEMM (i.e. a * A x B + b * C, use CUDA Dynamic Parallelism)
* matrix print (i.e PyTorch-style, adjustable edge elements)
* matrix console input (i.e. matrix[..., array[...)
* optional debug print level control (MMU_DEBUG)

