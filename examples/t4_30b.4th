\ Linear NN step-by-step verification
\ see https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
\ compile with ten4_config.h MM_DEBUG = 1
2 trace                         \ turn tracing on
1 1 2 1 nn.model                \ create our NN model
2 linear sigmoid                \ hidden layer
2 linear sigmoid                \ output layer
constant nn                     \ keep as a constant

nn                              \ fetch model
network                         \ show layers (non-destructive)

4 vector{ 0.15 0.2 0.25 0.3 }   \ update layer-0 weight, bias
0 nn.w=

2 vector{ 0.35 0.35 }
0 nn.b=

4 vector{ 0.4 0.45 0.5 0.55 }   \ update layer-2 weight, bias
2 nn.w=
2 vector{ 0.6 0.6 }
2 nn.b=

2 vector{ 0.05 0.1 }            \ input tensor
forward                         \ NN forward pass
." [0] linear input"   0 n@ .   \ L0 (layer-0) input i.e. 1st linear layer { 0.0500 0.1000 }
." [0] linear weight"  0 nn.w . \ L0 weight tensor { 0.15 0.2 0.25 0.3 }
." [0] linear bias"    0 nn.b . \ L0 bias tensor   { 0.35 0.35 }
." [1] sigmoid input"  1 n@ .   \ L1 input i.e. out0 = in0 @ wᵀ + b = { 0.3775, 0.3925 }
." [1] sigmoid filter" 1 nn.w . \ L1 filter s(1-s) = { 0.2413 0.2406 }
." [2] linear input"   2 n@ .   \ L2 input i.e outh1,h2 = { 0.5933 0.5969 }
." [3] sigmoid input"  3 n@ .   \ L3 linear input { 1.1059 1.2249 }
." [3] sigmoid filter" 3 nn.w . \ L3 filter s(1-s) = { 0.1868 0.1755 }
." [4] sigmoid output" 4 n@ .   \ L4 output layer { 0.7514 0.7729 }
." final output"      -1 n@ .   \ output from last layer (i.e. L4)

2 vector{ 0.01 0.99 }
constant tgt
." loss" tgt loss.mse .         \ verify loss= 0.596742

tgt backprop
." loss feedback" 4 n@    .     \ L4 dY={ 0.7414 -0.2172 }
." skip sigmoid"  3 n@    .     \ L3 dX={ 0.7414 -0.2172 }
." linear dB"     2 nn.db .     \ L2 dB=dY=(L3 dX)={ 0.7414 -0.2172 }
." linear dW"     2 nn.dw .     \ L2 dW=dYᵀ @ X
                                \      ={ { 0.7414 } { -0.2172 } } @ { 0.5933 0.5969 }
                                \      ={ { 0.4398 0.4425 } { -0.1288 -0.1296 }
." linear dX"     2 n@    .     \ L2 dX=dY @ W = { 0.1880 0.2142 }
." apply sigmoid" 1 n@    .     \ L1 dX={ 0.1880 0.2142 }
." linear dB"     0 nn.db .     \ L0 dB=dY={ 0.1880 0.2142 }
." linear dW"     0 nn.dw .     \ L0 dW   ={ { 0.0094 0.0188 } { 0.0107 0.0214 } }
." top layer dX"  0 n@    .     \ L0 dX=dB={ 0.0818 0.1019 }

0.5 nn.sgd                      \ learn at alpha=0.5 (default beta=0.0)
." L2 W"         2 nn.w .       \ L2 W={ { 0.3500 0.4000 } {  0.4500  0.5000 } }
                                \     - 0.5 * { { 0.4398 0.4425 } { -0.1289 -0.1296 } }
                                \     ={ { 0.1801 0.2287 } { 0.5644 0.6148 } }
." L2 dW"        2 nn.dw .      \ L2 dw=zeros (reset after sgd update)
." L2 B"         2 nn.b .       \ L2 b={ 0.6000 0.6000 } - 0.5 * { 0.7414 -0.2172 }
                                \     ={ 0.2293 0.7085 }
." L2 dB"        2 nn.db .      \ L3 db=zeros (reset after sgd update)
." L0 W"         0 nn.w         \ L0 w={ { 0.1500 0.2000 } { 0.2500 0.3000 }
                                \      - 0.5 * { 0.0094 0.0188 } { 0.0107 0.0214 }
                                \     ={ { 0.1453 0.1906 } { 0.2446 0.2893 } }
.( verify L0 W={ { 0.1453 0.1906 } { 0.2446 0.2893 } } => ) .
." L0 B"          0 nn.b        \ L0 b= { 0.3500 0.3500 } - 0.5 * { 0.1880 0.2142 }
                                \     = { 0.2560 0.2429 }
.( verify L0 B={ 0.2560 0.2429 } => ) .
bye

