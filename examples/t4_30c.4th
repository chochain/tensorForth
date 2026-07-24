\ Linear NN step-by-step with 2 samples verification
\ see https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
\ compile with ten4_config.h MM_DEBUG = 1
2 trace                         \ turn tracing on
2 1 2 1 nn.model                \ create our NN model
2 linear sigmoid                \ hidden layer
2 linear sigmoid                \ output layer
constant nn                     \ keep as a constant

nn                              \ fetch model
network                         \ show layers

4 vector{ 0.15 0.2 0.25 0.3 }   \ update layer[0] weight, bias
0 nn.w=
2 vector{ 0.35 0.35 }
0 nn.b=

4 vector{ 0.4 0.45 0.5 0.55 }   \ update layer[2] weight, bias
2 nn.w=
2 vector{ 0.6 0.6 }
2 nn.b=

4 vector{ 0.05 0.1 0.05 0.1 }   \ create input vector (auto reshaped => 2 1 2 1 tensor)
forward                         \ NN forward pass
." [0] linear input"   0 n@ .   \ L0 (layer-0) input i.e. 1st linear layer { 0.0500 0.1000 }x2
." [0] linear weight"  0 nn.w . \ L0 weight tensor { 0.15 0.2 0.25 0.3 }
." [0] linear bias"    0 nn.b . \ L0 bias tensor   { 0.35 0.35 }
." [1] sigmoid input"  1 n@ .   \ L1 input i.e. out0 = in0 @ wᵀ + b = { 0.3775, 0.3925 }x2
." [1] sigmoid filter" 1 nn.w . \ L1 filter s(1-s) = { 0.2413 0.2406 }x2
." [2] linear input"   2 n@ .   \ L2 input i.e outh1,h2 = { 0.5933 0.5969 }x2
." [3] sigmoid input"  3 n@ .   \ L3 linear input { 1.1059 1.2249 }x2
." [3] sigmoid filter" 3 nn.w . \ L3 filter s(1-s) = { 0.1868 0.1755 }x2
." [4] sigmoid output" 4 n@ .   \ L4 output layer { 0.7514 0.7729 }x2
." final output"      -1 n@ .   \ output from last layer (i.e. L4)

4 vector{ 0.01 0.99 0.01 0.99 } \ create target vector
2 1 2 1 reshape4                \ => 2 1 2 1 tensor (matching output, no auto reshape)
constant tgt
." loss" tgt loss.mse .         \ loss= 0.596742 (avg, should be the same as N=1)

tgt backprop                    \ back propegation
." loss feedback" 4 n@    .     \ L4 dY={ 0.7414 -0.2172 }x2
." skip sigmoid"  3 n@    .     \ L3 dX={ 0.7414 -0.2172 }x2
." linear dB"     2 nn.db .     \ L2 dB=dY=(L3 ΣdX)={ 1.4827 -0.4341 }
." linear dW"     2 nn.dw .     \ L2 dW=Σ(dYᵀ @ X)
                                \      ={ { 0.7414 } { -0.2172 } } @ { 0.5933 0.5969 }
                                \      ={ { 0.8797 0.8850 } { -0.2576 -0.2591 }
." apply sigmoid" 1 n@    .     \ L1 dX={ 0.1880 0.2142 }x2
." linear dB"     0 nn.db       \ L0 dB=dY={ 0.3760 0.4284 }
.( verify db = { 0.3760 0.4284 } => ) .
." linear dW"     0 nn.dw       \ L0 dW   ={ { 0.0188 0.0376 } { 0.0214 0.0428 } }
.( verify dw = { { 0.0188 0.0376 } { 0.0214 0.0428 } } => ) .
." top layer dX"  0 n@          \ L0 dX=dB={ 0.0818 0.1019 }x2
.( verify n@ { { { 0.0818 } { 0.1019 } } { { 0.0818 } { 0.1019 } } } => ) .

0.5 0.0 nn.sgd                  \ SGD learn at alpha=0.5, beta 0.0 (default beta=0.9)
." L2 W"         2 nn.w .       \ L2 W={ { 0.3500 0.4000 } {  0.4500  0.5000 } }
                                \     - 0.5 * { { 0.8797 0.8850 } { -0.2576 -0.2591 } }
                                \     ={ { -0.0398 0.0075 } { 0.6288 0.6796 } }
." L2 dW"        2 nn.dw .      \ L2 dw=zeros (reset after sgd update)
." L2 B"         2 nn.b .       \ L2 b={ 0.6000 0.6000 } - 0.5 * { 1.4827 -0.4341 }
                                \     ={ -0.1414 0.8171 }
." L2 dB"        2 nn.db .      \ L3 db=zeros (reset after sgd update)
." L0 W"         0 nn.w         \ L0 w={ { 0.1500 0.2000 } { 0.2500 0.3000 }
                                \      - 0.5 * { 0.0188 0.0376 } { 0.0214 0.0428 }
                                \     ={ { 0.1406 0.1812 } { 0.2393 0.2786 } }
.( verify L0 W={ { 0.1406 0.1812 } { 0.2393 0.2786 } } => ) .
." L0 B"          0 nn.b        \ L0 b= { 0.3500 0.3500 } - 0.5 * { 0.3760 0.4284 }
                                \     = { 0.1620 0.1358 }
.( verify L0 B={ 0.1620 0.1358 } => ) .
bye

