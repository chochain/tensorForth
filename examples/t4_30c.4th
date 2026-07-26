\ Linear NN step-by-step with 3 samples verification
\ see https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
0 trace                         \ turn off tracing, default 1
3 1 2 1 nn.model                \ create our NN model
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

6 vector{ 0.05 0.1 0.05 0.1 0.05 0.1 }   \ create input vector (auto reshaped => 3 1 2 1 tensor)
forward                         \ NN forward pass
." L0 linear input="   0 n@ .   \ L0 (layer-0) input i.e. 1st linear layer { 0.0500 0.1000 }x3
." L0 linear weight="  0 nn.w . \ L0 weight tensor { 0.15 0.2 0.25 0.3 }
." L0 linear bias="    0 nn.b . \ L0 bias tensor   { 0.35 0.35 }
." L1 sigmoid input="  1 n@ .   \ L1 input i.e. out0 = in0 @ wᵀ + b = { 0.3775, 0.3925 }x3
." L1 sigmoid filter=" 1 nn.w . \ L1 filter s(1-s) = { 0.2413 0.2406 }x3
." L2 linear input="   2 n@ .   \ L2 input i.e outh1,h2 = { 0.5933 0.5969 }x3
." L3 sigmoid input="  3 n@ .   \ L3 linear input { 1.1059 1.2249 }x3
." L3 sigmoid filter=" 3 nn.w . \ L3 filter s(1-s) = { 0.1868 0.1755 }x3
." L4 sigmoid output=" 4 n@ .   \ L4 output layer { 0.7514 0.7729 }x3
." final output="      -1 n@ .  \ output from last layer (i.e. L4)

6 vector{ 0.01 0.99 0.01 0.99 0.01 0.99 } \ create target vector
3 1 2 1 reshape4                \ => 2 1 2 1 tensor (matching output, no auto reshape)
constant tgt
tgt loss.mse                    \ loss= 0.596742 (avg, should be the same as N=1)
." verify loss=0.596742=> " .

tgt backprop                    \ back propegation
." L4 loss feedback=" 4 n@    . \ L4 dY={ 0.7414 -0.2172 }x3
." L3 skip sigmoid="  3 n@    . \ L3 dX={ 0.7414 -0.2172 }x3
." L2 linear dB="     2 nn.db . \ L2 dB=dY=(L3 ΣdX)={ 2.2241 -0.6512 }
." L2 linear dW="     2 nn.dw . \ L2 dW=Σ(dYᵀ @ X)
                                \      ={ { 0.7414 } { -0.2172 } } @ { 0.5933 0.5969 }x3
                                \      ={ { 1.3195 1.3275 } { -0.3836 -0.3887 } }
." L1 apply sigmoid=" 1 n@    . \ L1 dX={ 0.1880 0.2142 }x3
." L0 linear dB="     0 nn.db   \ L0 dB=dY={ 0.5640 0.6427 }
." verify db = { +0.5640 +0.6427 } => " .
." L0 linear dW="     0 nn.dw   \ L0 dW   ={ { 0.0288 0.0564 } { 0.0321 0.0643 } }
." verify dw = { { +0.0282 +0.0564 } { +0.0321 +0.0643 } } => " .
." L0 top layer dX="  0 n@      \ L0 dX=dB={ 0.0818 0.1019 }x3
." verify n@ { { { +0.0818 +0.1019 } }x3 } => " .

0.5 0.0 nn.sgd                  \ SGD learn at alpha=0.5, beta 0.0 (default beta=0.9)
." L2 W="         2 nn.w .      \ L2 W={ { 0.3500 0.4000 } {  0.4500  0.5000 } }
                                \     - 0.5 * { { 1.3195 1.3275 } { -0.3836 -0.3887 } }
                                \     ={ { -0.2597 -0.2138 } { 0.6932 0.7443 } }
." L2 dW="        2 nn.dw .     \ L2 dw=zeros (reset after sgd update)
." L2 B="         2 nn.b .      \ L2 b={ 0.6000 0.6000 } - 0.5 * { 2.2241 -0.6512 }
                                \     ={ -0.5120 0.9256 }
." L2 dB="        2 nn.db .     \ L3 db=zeros (reset after sgd update)
." L0 W="         0 nn.w        \ L0 w={ { 0.1500 0.2000 } { 0.2500 0.3000 }
                                \      - 0.5 * { 0.0288 0.0564 } { 0.0321 0.0643 }
                                \     ={ { 0.1359 0.1718 } { 0.2339 0.2679 } }
." verify L0 W={ { +0.1359 +0.1718 } { +0.2339 +0.2679 } } => " .
." L0 B="         0 nn.b        \ L0 b= { 0.3500 0.3500 } - 0.5 * { 0.5640 0.6427 }
                                \     = { 0.0680 0.0287 }
." verify L0 B={ +0.0680 +0.0287 } => " .
bye

