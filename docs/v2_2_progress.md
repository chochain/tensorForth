# tensorForth - Release 2.2 / 2022-09
## Features
  * KV   new data structure to keep kv pair (i.e. associative array)
  * CNN
    + define word as the functional (implicit sequential)
    + words are destructive by default (i.e. input tensor updated)

### CNN Example
<pre>
: seq1 (Ta -- Ta') 10 5 conv2 2 maxpool relu ;
: seq2 (Ta -- Ta') 20 5 conv2 0.5 dropout2 2 maxpool relu ;
: lin1 (Ta -- Ta') -1 320 reshape relu 50 linear ;
: lin2 (Ta -- Ta') 0.5 dropout 10 linear ;
: forward (Ta -- Ta') seq1 seq2 lin1 lin2 softmax ;
: train (set0 -- ) for_batch autograd forward loss.nll backward 0.1 0.9 sgd next nn.save ;
: test  (set1 -- ) for_batch forward loss.nll avg predict next ;
</pre>

### CNN volcabularies
#### Tensor ops
  * sum (Ta -- n) - array, matrix sum (with prefix sum or reduce)
  * stack         - joining arrays (on given axis)
  * split         - split an array into multi subarrays
  * reshape with negative value (i.e. inferred size = size / other dims)

#### Load/Save - .npy
  * nn.dir       ( -- )      - list dataset directory
  * nn.load root ( -- Ta Tb) - test_dataset train_dataset
    - Ta[10000,28,28,1]
    - Tb[50000,28,28,1]
  * nn.save      (Ta -- )    - save tensor/gradiant
    
#### Conv2D
  * conv2 (Ta c   -- Ta') - c channel output, 3x3 filter, stride=1, padding=0, dilation=0, bias=0.5
  * conv2 (Ta c f -- Ta') - c output, fxf filter, stride
  * conv2 (Ta c A -- Ta') - n output, array[3, 1, 0, 1, 0.5] for (3x3, stride=1, padding=0, dilation=1, bais=0.3)

#### Activation (non-linear)
  * relu    (Ta -- Ta')
  * sigmoid (Ta -- Ta')
  * softmax (Ta -- Ta')
    
#### Pooling
  * maxpool (Aa   -- n)   - on 1D array (RNN?)
  * minpool (Aa   -- n)   - on 1D array
  * avgpool (Aa   -- n)   - 
  * avgpool (Ta   -- Ta') - 2x2
  * avgpool (Ta n -- Ta') - nxn
  * maxpool (Ta   -- Ta') - 2x2
  * maxpool (Ta n -- Ta') - nxn
  * minpool (Ta   -- Ta') - 2x2
  * minpool (Ta n -- Ta') - nxn
  
#### Linear (fully connected)
  * linear  (Ta n -- Ta') - linearize (y = Wx + b) from Ta input to n out_features

#### Dropout (reduce strong correlation, improve feature independence)
  * dropout  (Ta p -- Ta') zero out p% of channel data (add noise between data points)
  * dropout2 (Ta p -- Ta') zero out p% of input tensor elements (add noise between pixels), usually used after conv2d

#### Loss
  * loss.nll (Ta Tb -- Tc) - negative likelihood
  * loss.mse (Ta Tb -- Tc) - mean sqare error
  * loss.ce  (Ta Tb -- Tc) - cross-entropy  
  * predict  (Ta    -- n)  - cost function (avg all losts)

#### Back Propergation
  * autograd ( -- )          - enable Direct Acylic Graph building with zero gradiant
  * backward (Ga -- Ga Ta)   - stop DAG building, compute gradiant
  * sgd      (Ga Ta p m -- ) - apply SGD(learn_rate=p, momentum=m) backprop on DAG
  * adam     (Ga Ta p m -- ) - apply Adam backprop
    

