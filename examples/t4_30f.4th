.( ## MNIST convolution to TensorBoard output ## ) cr
256 constant N                      \ mini-batch size (number of samples)
0 value t0                          \ starting time (benchmark)
1 value dsz0 1 value dsz1           \ number of corpus sample
variable hit                        \ create var for hit counter, and zero it
variable lox                        \ create var for epoch latest loss
0.001 value lr                      \ init learning rate (for Adam)

.( ### our network model )
N 28 28 1 nn.model                  \ create a model (100 per mini-batch of 28x28x1 img)
0.5 10 conv2d 2 maxpool relu        \ 2D convolution layer (10 output channels, bias=0.5)
flatten 100 linear relu             \ a linear layer with relu (100 channels)
10 linear softmax                   \ 2nd linear layer (10 channels) and softmax output
constant md0                        \ keep as a constant

.( ### statistics and histogram routines )
: stat ( n -- )                     \ statistics sent to tensorboard
  dup ." epoch=" .  dup .tbstep     \ set tensorboard step (keep copy)
  clock t0 - 1000 / dup dup s" train/time" .scalar \ time (in sec)
  . ." sec" hit @   dup ."  hit=" . cr             \ hit per epoch
  dsz0 /            dup s" train/acc"  .scalar     \ accuracy
  lox @             dup s" train/loss" .scalar     \ loss
  lr                dup s" train/lr"   .scalar     \ learn rate
  s" MNIST step=%d, time=%g, acc=%g loss=%g learn_rate=%g" sprintf \ text substitude
  s" progress/text" .text ;
  
: histo ( M -- M )                  \ capture histogram to tensorboard
  0 nn.w 30 s" nn/conv0" .histo       \ convolution filter (30-buckets)
  2 nn.w 30 s" nn/relu2" .histo       \ activation 
  4 nn.w 30 s" nn/lin4"  .histo       \ 1st linear filter
  6 nn.w 30 s" nn/lin6"  .histo ;     \ 2nd linear filter
  
.( ### setup datasets )
N dataset mnist_train               \ create MNIST dataset with model batch size
nn.len to dsz0                      \ get dataset total number of samples
constant ds0                        \ keep dataset in a constant

N dataset mnist_test                \ create MNIST test dataset with model batch size
nn.len to dsz1                      \ get dataset total number of samples
constant ds1                        \ keep testing dataset as a constant

ds0 16 s" mnist/train" .tile        \ sample training dataset, 16-wide, to tensorboard
ds1 16 s" mnist/test"  .tile        \ sample testing  dataset, 16-wide

.( ### create our CNN framework )
: train_epoch ( M -- M' )           \ one epoch of trainning i.e. to learn
  0 hit ! ds0 rewind                  \ run thru trainning dataset
  for                                 \ starting first mini-batch (from return stack)
    forward                           \ neural network forward pass
    loss.ce lox ! nn.hit hit +!       \ collect latest loss and accumulate hit
    backprop                          \ neural network back propegation
    lr nn.adam                        \ train with Adam Gradient Descent (b1=0.9,b2=0.999)
  next ;                              \ fetch next mini-batch from return stack (till done)

: test_epoch ( M -- M )             \ one epoch of validation, i.e. to check how well
  0 hit ! ds1 rewind                  \ run thru testing dataset
  for                                 \ starting first mini-batch (from return stack)
     forward                          \ forward pass
     nn.hit hit +!                    \ collect latest accumulate hit
  next                                \ fetch next mini-batch
  hit @ dsz1 / dup ." test/acc=" . cr \ show test accuracy
  s" test/acc" .scalar ;              \ send to tensorboard

: cnn ( M n -- M' )                 \ full CNN run
  clock [to] t0                       \ get starting time
  1+ 0 do                             \ multiple epochs [0..n]
    train_epoch                       \ run one trainning epoch
    r@ stat histo                     \ send statistics, histogram to tensorboard
    test_epoch                        \ run one validation epoch
    lr 0.9 * [to] lr                  \ decay learning rate
  loop ;

0 trace
.( ###  )
md0 network dup .graph              \ put model as TOS, show and to tensorboard graph
20 cnn                              \ execute multiple (20) epoches

bye
