# tensorForth - Release 2.2 / 2022-09
## Features
  * CNN
    + define word as the functional (implicit sequential)
    + words are destructive by default (i.e. input tensor updated)

### CNN volcabularies
#### Load - .npy
  * load root ( -- Ta Tb) - test_dataset train_dataset
    - Ta[10000,28,28,1]
    - Tb[50000,28,28,1]
    
#### Conv2D
  * conv2 (Ta   -- Ta') - default 3x3 filter, stride=1, padding=0, dilation=0, bias=0.5
  * conv2 (Ta n -- Ta') - nxn filter
  * conv2 (Ta A -- Ta') - array[3, 1, 0, 1, 0.5] for (3x3, stride=1, padding=0, dilation=1, bais=0.3)

#### Activation (non-linear)
  * relu    (Ta -- Ta')
  * sigmoid (Ta -- Ta')
  * softmax (Ta -- Ta')
    
#### Pooling
  * max (Aa   -- n)   - on 1D array (RNN?)
  * min (Aa   -- n)   - on 1D array
  * avg (Aa   -- n)   - 
  * avg (Ta   -- Ta') - 2x2
  * avg (Ta n -- Ta') - nxn
  * max (Ta   -- Ta') - 2x2
  * max (Ta n -- Ta') - nxn
  * min (Ta   -- Ta') - 2x2
  * min (Ta n -- Ta') - nxn
  
#### Linear (fully connected)
  * linear => word: (y=Wx+b)

#### Dropout
  * dropout => word:

#### Loss
  * sum      (Ta    -- n)  - matrix sum (with prefix sum or reduce)
  * loss.nll (Ta Tb -- Tc) - negative likelihood
  * loss.mse (Ta Tb -- Tc) - mean sqare error
  * loss.ce  (Ta Tb -- Tc) - cross-entropy  
  * cost     (Ta    -- n)  - cost function (avg all losts)

#### Optimizer
  * sgd      - stochastic gradiant decent
  * adam     - 

#### Back Propergation
  * backward - back propergation
    

