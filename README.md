## cueForth - CUDA eForth
* Forth VMs that support dynamic parallelism

### TODO
* add tensor object
* integrate CUTLASS

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

