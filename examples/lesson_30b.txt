\ Linear NN step-by-step verification
\ see https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
\ compile with ten4_config.h MM_DEBUG = 1
1 trace                         \ turn tracing on
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
-1 n@ .                         \ verify output { 0.7514 0.7729 }
3 nn.w .                        \ verify sigmoid filter s(1-s) = { 0.1868 0.1755 }
2 n@ .                          \ verify layer-2 output i.e outh1,h2 = { 0.5933 0.5969 }
1 nn.w .                        \ sigmoid filter s(1-s) = { 0.2413 0.2406 }

2 vector{ 0.01 0.99 }
constant tgt
tgt loss.mse .                  \ verify loss= 0.596742

tgt backprop
3 n@    .                       \ verify L3 dX={ 0.1385 -0.0381 }
2 nn.db .                       \ confirm db = dX
2 nn.dw .                       \ verify L2 dw={ { 0.0822 0.0827 } { -0.0226 -0.0227 } }
1 n@    .                       \ verify L1 dX={ 0.0088 0.0100 }
0 nn.db .                       \ confirm db = dX
0 nn.dw .                       \ verify L1 dw={ { 0.0004 0.0009 } { 0.0005 0.0010 } }

0.5 nn.sgd                      \ learn at alpha=0.5
2 nn.w .                        \ verify L2 w={ { 0.3589 0.4087 } { 0.5113 0.5614 } }
2 nn.b .                        \ verify L2 b={ 0.5308 0.6190 }
0 nn.w                          \ verify L0 w={ { 0.1498 0.1996 } { 0.2498 0.2995 } }
.( verify { { 0.1498 0.1996 } { 0.2498 0.2995 } } => ) .
0 nn.b                          \ verify L0 b={ 0.3456 0.3450 }
.( verify { 0.3456 0.3450 } => ) .
bye

