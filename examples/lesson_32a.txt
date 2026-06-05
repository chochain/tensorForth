.( ## GAN tests Z = X @ A + B ) cr      \ linear regression in matrix form
64 constant N                           \ mini-batch size (64 points)

.( ### regression matrix and offset )
2 2 matrix{ 1 2 -0.1 0.5 } constant A   \ Gaussian transformation matrix
1 2 matrix{ 1 2 }          constant B   \ create offset matrix
: X ( -- t4 ) N 1 2 1 tensor randn ;    \ N points of random { x1, x2 }
: Z ( -- t4 ) X A @= B += ;             \ one set of real samples i.e. Z = AX + B

: w_ ( N -- N' )                        \ init w to 0.02 (~0.707 too aggressive)
  -2 nn.w 0.02 fill drop                \ get w of last layer (-1 is output), set to 0.02
  -2 nn.b 0.02 fill drop ;              \ get b of last layer, set to 0.02
  
.( ### build generator network )
N 1 2 1 nn.model                        \ generator model
2 linear w_                             \ just one linear layer, w=2x2 (bias=1.0)
flatten                                 \ trainable (linear + MSE => pass-thru)
constant G                              \ kept as a constant

.( ### build discriminator network )
N 1 2 1 nn.model                        \ discriminator model
5 linear w_ 0.2 leakyrelu               \ 1st linear layer
3 linear w_ 0.2 leakyrelu               \ 2nd linear layer
1 linear w_ sigmoid                     \ binary output layer
constant D                              \ kept as a constant

.( ### statistics and weight/bias dump )
0 value _g 0 value _r 0 value _f                  \ loss for gen, real, and fake
: stat ( -- )                                     \ display statistics
  cr ." w,b=" G 0 nn.w . 0 nn.b . drop
  ." G=" _g . ." , Dr=" _r . ." , Df=" _f . cr ;

.( ### our entire GAN here )
N 1 1 1 tensor ones  constant REAL                \ onehot tersor for a real set
N 1 1 1 tensor zeros constant FAKE                \ onehot tensor for a fake set
: F ( -- t4 ) G X forward -1 n@ swap drop ;       \ generate a mini-batch of fake samples
: train_d ( D -- D' )
  1 trainable                                     \ make D discriminator trainable
  Z forward REAL loss.bce [to] _r REAL backprop   \ treat real samples as real
  F forward FAKE loss.bce [to] _f FAKE backprop   \ treat fake samples as fake
  0.001 nn.adam ;                                 \ train, Adam (b1=0.9,b2=0.999)
: train_g ( D -- D' )
  0 trainable                                     \ make D testing mode (read only)
  F forward REAL loss.bce [to] _g REAL backprop   \ now treat fake samples as real
  0 n@ G swap ( D G t ) backprop                  \ propagate dX back to G
  0.001 nn.adam ( D G ) drop ;                    \ refine/train G with Adam

: epoch ( -- ) D                                  \ put D on TOS
  40 for train_d train_g next                     \ train with 40 * N samples
  drop ;                                          \ drop D
: gan ( n -- ) 1-                                 \ run n epoch
  for
    epoch stat
    F r@ s" e%d" sprintf .embed                   \ send to tensorboard (projector)
  next ;

.( ### expect 50% loss G, Dr, Df ~> 0.69 = ln 0.5, can't tell the difference)
0 trace
Z s" z0" .embed                                   \ send real set to tensorboard 
20 gan                                            \ run multiple (20) epochs

bye
