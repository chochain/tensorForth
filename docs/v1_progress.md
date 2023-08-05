# tensorForth - Release 1.0 / 2022-05
## Features
* Dr. Ting's eForth words with F32 as data unit, U16 instruction unit
* Support parallel Forth VMs
* Lambda-based Forth microcode
* Memory management unit handles dictionary, stack, and parameter blocks in CUDA
* Managed memory debug utilities, words, see, ss_dump, mem_dump
* String handling utilities in CUDA
* Light-weight vector class, no dependency on STL
* Output Stream, async from GPU to host

### Initialization
|stage|snap|
|---|---|
|begin|<img src="./img/cueforth_init_0.png">|
|end|<img src="./img/cueforth_init_1.png">|

### Outer Interpreter
<img src="./img/cueforth_words_0.png">

### Test - Dr. Ting's eForth lessons
|case#|ok|snap|
|---|---|---|
|repeat|pass|<img src="./img/cueforth_ast_0.png">|
|weather|pass|<img src="./img/cueforth_weather_0.png">|
|multiply|pass|<img src="./img/cueforth_mult_0.png">|
|calendar|pass|<img src="./img/cueforth_calndr_1.png">|

