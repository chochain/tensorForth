# tensorForth - Release 4.0 / 2026-06

The CUDA Dynamic Parallelism v2.0 is not backward compatible. Before v4.0, tensorForth depends heavily on the synchronization behaviour of GPUs. Unfortunately, nVidia has decided the cost of keeping track of internal synchronization was too high, the GPU cores became asynchronous after Hopper, i.e. CUDA SM >= 9.0 . Turning and Ampere still work but are dangling. So, **tensorForth** is forced to be redesigned and rebuilt from ground up.

## Architecture Changes

Instead of having VMs, compute tasks both run on GPU, only passing IO out to the host, today, the architecture of **tensorForth** has VMs, IO run on host and only send computation-heavy tasks to GPU.

<pre>
tensorForth
o------ VM[] (VM pool)
|       |
|       v
o------ System (System Interface, singleton)
        o-- _istr (input stream)
        o-- _ostr (output stream)
        o-- _tib  (input buffer)
        |
        +--> MMU (Memory Management Unit, singleton)
        |    o-- _mpool    (object metadata store)
        |    o-- _ostore   (object content store)
        |    o-- _dict     (Dictionary)
        |    o-- _vmss     (VM data stacks)
        |    o-- _vmrs     (VM return stacks)
        |    o-- _pmem     (parameter memory block)
        |
        +--> AIO (Async IO, singleton)
        |    o-- istream   (input stream)
        |    o-- ostream   (output stream)
        |    o-- ifstream  (file inupt, model load)
        |    o-- ofstream  (file output, model save)
        |
        +--> Summary (TensorBoard Event Summary Writer)
        |    o--> png
        |    o--> crc32c
        |    +--> Projector
        |    |    o-- ofstream (graph file output)
        |    +--> Encoder
        |         o-- ofstream (event file output)
        |
        +--> Debug (friend object for debuging/tracing)
</pre>

### Components, Memory Allocation, and Sizing

|Component|Memory|Count/Size|
|---|---|---|
|VM Pool|host|1|
|VM|host|T4_VM_COUNT|
|Dictionary|host|T4_DICT_SZ|
|Data Stack|host|T4_VM_COUNT * T4_SS_SZ|
|Return Stack|host|T4_VM_COUNT * T4_SS_SZ|
|Parameter Memory|host|T4_PMEM_SZ|
|Object Store|device|T4_OSTORE_SZ|

* Note on Word (lambda)|host|dynamically allocated|
* Note on SS, RS using __shared memory

### Core Object Types
<pre>
                  T4Base
                    ^
                    |
       +------------+-----------+
       |            |           |
     Tensor       Model      Dataset
</pre>

### Forth Virtual Machine
<pre>
                    VM      base class
                    ^
                    |
                 ForthVM    eForth core processor (i.g. dup, drop)
                    ^
                    |
                 TensorVM   tensor operation processor (i.g. blas, gemm)
                    ^
                    |
                  NetVM     machine learning processor (i.g. forward, backprop, adam)
</pre>

Each Forth VM, from VM Pool, manages its own states

<pre>
  o-- state               (STOP, HOLD, QUERY, NEST)
  o-- ip                  (pointer to parameter memory)
  o-- tos                 (cached top of data stack)
  o-- data stack pointer
  o-- return stack pointer
</pre>

Also, each VM operates its own outer-interpreter which processes incoming stream continuously similar to that of an OS event loop processor. As the stream is processed, words are processed one-by-one and dispatched. While classic Forth words are handled on CPU, math-heavy words are send to GPUs utilizing their massively parallel cores.

### Memory Pools - Two Class of Allocators

* On host   - metadata of objects (Tensors, Model, Dataset), fixed-size
* On device - the content of objects, CUDA Unified Memory, TLSF-based

### Updates (from TODO)
* VM
  + CUDA 12 migration
* Design & Instrumentation
  + Visulization via TensorBoard
    - output tensor in HWC format
    - tiled png output
    - export raw image to png (with STB), and for PIL (Python Image Lib), matplotlib
* Tuning
  + Use nVidia intrinsic functions when available
  + Review all BLAS kernel functions, collected into t4math module
    - use grid-stride and warp shuffle when possible
    - unroll every loop when possible
    - highly tuned GEMM, shared memory tile 
    - refined Gauss-Jordan/LU matrix ops
  + Review all Neural network kernel functions

### References
+ [GEMM](https://siboehm.com/articles/22/CUDA-MMM)
+ [tfevents](https://github.com/mlverse/tfevents)
+ [protobuf](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/how_tos/tool_developers/index.md)
+ [tensorflow EventWriter](https://stackoverflow.com/questions/48610803/how-i-can-use-filewrite-summary-in-tensorflow-c-api-to-view-it-in-tensorboard/48702823#48702823)
+ [pytorch SummaryWriter](https://github.com/pytorch/pytorch/blob/main/torch/utils/tensorboard/writer.py)
+ [pytorch TensorBoard writer](https://github.com/pytorch/pytorch/blob/main/torch/utils/tensorboard/writer.py)



