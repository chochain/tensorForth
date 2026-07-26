\ Linear NN step-by-step verification
\ see https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
1 1 2 1 nn.model                \ create our NN model
3 linear sigmoid                \ hidden layer
2 linear sigmoid                \ output layer
constant nn                     \ keep as a constant

nn                              \ fetch model
network                         \ show layers (non-destructive)

6 vector{ 0.15 0.2 0.25 0.3 0.2 0.15 }   \ update layer-0 weight, bias
0 nn.w=

3 vector{ 0.35 0.35 0.35 }
0 nn.b=

6 vector{ 0.4 0.45 0.5 0.55 0.5 0.45 }   \ update layer-2 weight, bias
2 nn.w=
2 vector{ 0.6 0.6 }
2 nn.b=

2 vector{ 0.05 0.1 }            \ input tensor
forward                         \ NN forward pass
." L0 linear input="   0 n@ .   \ L0 (layer-0) input i.e. 1st linear layer { 0.0500 0.1000 }
." L0 linear weight="  0 nn.w . \ L0 weight tensor { 0.15 0.2 0.25 0.3 0.2 0.15 }
." L0 linear bias="    0 nn.b . \ L0 bias tensor   { 0.35 0.35 0.35 }
." L1 sigmoid input="  1 n@ .   \ L1 input i.e. out0 = in0 @ wᵀ + b = { 0.3775, 0.3925 0.3750 }
." L1 sigmoid filter=" 1 nn.w . \ L1 filter s(1-s) = { 0.2413 0.2406 0.2414 }
." L2 linear input="   2 n@ .   \ L2 input i.e outh1,h2 = { 0.5933 0.5969 0.5927 }
." L3 sigmoid input="  3 n@ .   \ L3 linear input { 1.4022 1.4914 }
." L3 sigmoid filter=" 3 nn.w . \ L3 filter s(1-s) = { 0.1585 0.1500 }
." L4 sigmoid output=" 4 n@ .   \ L4 output layer { 0.8025 0.8163 }
." final output="     -1 n@ .   \ output from last layer (i.e. L4)

2 vector{ 0.01 0.99 }
constant tgt
tgt loss.mse                    \ verify loss= 0.658292
." verify loss=0.658292=> " .

tgt backprop
." L4 loss feedback=" 4 n@    . \ L4 dY={ 0.7925 -0.1737 }
." L3 skip sigmoid="  3 n@    . \ L3 dX={ 0.7925 -0.1737 }
." L2 linear dB="     2 nn.db . \ L2 dB=dY=(L3 dX)={ 0.7925 -0.1737 }
." L2 linear dW="     2 nn.dw . \ L2 dW=dYᵀ @ X
                                \      ={ { 0.7925 } { -0.1737 } } @ { 0.5933 0.5969 0.5927 }
                                \      ={ { 0.4702 0.4731 0.4697 } { -0.1031 -0.1037 -0.1029 } }
." L2 linear dX="     2 n@    . \ L2 dX=dY @ W = { 0.2215 0.2698 0.3181 }
." L1 apply sigmoid=" 1 n@    . \ L1 dX={ 0.2215 0.2698 0.3181 }
." L0 linear dB="     0 nn.db . \ L0 dB=dY={ 0.2215 0.2698 0.3181 }
." L0 linear dW="     0 nn.dw . \ L0 dW={ { 0.2215 } { 0.2698 } { 0.3181 } } @ { 0.05 0.10 }
                                \      ={ { 0.0111 0.0221 } { 0.0135 0.0270 } { 0.0159 0.0318 } }
." L0 top layer dX="  0 n@    . \ L0 dX={ 0.2215 0.2698 0.3181 } @ { { 0.15 0.2 } { 0.25 0.3 } { 0.2 0.15 } }
                                \      ={ 0.1643 0.1729 }

0.5 0.0 nn.sgd                  \ SGD learn at alpha=0.5, beta 0.0 (default beta=0.9)
." L2 W="         2 nn.w .      \ L2 W={ { 0.4000 0.4500 } {  0.5000  0.5500 } { 0.5000 0.4500 } }
                                \     - 0.5 * { { 0.4702 0.4731 0.4697 } { -0.1031 -0.1037 -0.1029 } }
                                \     ={ { 0.1649 0.2135 0.2651 } { 0.6015 0.5518 0.5015 } }
." L2 dW="        2 nn.dw .     \ L2 dw=zeros (reset after sgd update)
." L2 B="         2 nn.b .      \ L2 b={ 0.6000 0.6000 } - 0.5 * { 0.7925 -0.1737 }
                                \     ={ 0.2037 0.6869 }
." L2 dB="        2 nn.db .     \ L3 db=zeros (reset after sgd update)
." L0 W="         0 nn.w        \ L0 w={ { 0.1500 0.2000 } { 0.2500 0.3000 } { 0.2000 0.1500 }
                                \      - 0.5 * { { 0.0111 0.0221 } { 0.0135 0.0270 } { 0.0159 0.0318 } }
                                \     ={ { 0.1445 0.1889 } { 0.2433 0.2865 } { 0.1920 0.1341 } }
." verify L0 W={ { +0.1445 +0.1889 } { +0.2433 +0.2865 } { +0.1920 +0.1341 } } } => " .
." L0 B"          0 nn.b        \ L0 b= { 0.3500 0.3500 0.3500 } - 0.5 * { 0.2215 0.2698 0.3181 }
                                \     = { 0.2393 0.2151 0.1909 }
." verify L0 B={ +0.2393 +0.2151 +0.1909 } => " .
bye

