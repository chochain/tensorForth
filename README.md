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

### TODO
* add tensor object (study torch tensor, 128-bit NHWC tensor)
* integrate CUB, CUTLASS (utilities.init, gemm_api)
* formatted file IO (CSV, Numpy)
* ML cases and benchmark (kaggle.MNIST, ...)
* add inter-VM communication (CUDA stream)
* add dynamic graph (GNN)
* integrate plots (tensorboard, R)

### Progress
#### Initialization
|stage|snap|
|---|---|
|begin|<img src="./img/cueforth_init_0.png">|
|end|<img src="./img/cueforth_init_1.png">|

#### Outer Interpreter
<img src="./img/cueforth_words_0.png">

#### Test - Dr. Ting's eForth lessons
|case#|ok|snap|
|---|---|---|
|repeat|pass|<img src="./img/cueforth_ast_0.png">|
|weather|pass|<img src="./img/cueforth_weather_0.png">|
|multiply|pass|<img src="./img/cueforth_mult_0.png">|
|calendar|pass|<img src="./img/cueforth_calndr_1.png">|

