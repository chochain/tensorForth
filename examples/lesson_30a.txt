\ NN linear layer verification
\ compile with ten4_config MM_DEBUG = 1
1 1 2 1 nn.model                 \ create a one-layer neural network model
3 linear                         \ add layer[0], a 2x3 fully connected 
constant nn                      \ keep in a constant

3 2 matrix{ 1 2 3 4 5 6 }        \ create weight matrix
0.1 *=                           \ reduce to 1/10
constant w                       \ keep in a constant

3 vector{ 1 2 3 }                \ create bias vector
constant b                       \ keep in a constant

nn                               \ fetch the network model 
0 nn.w .                         \ show layer[0] weight parameters
0 nn.b .                         \ show layer[0] bias parameters 

w 0 nn.w=                        \ set layer[0] weight parameters
0 nn.w                           \ verify { { 0.1 0.2 } { 0.3 0.4 } { 0.5 0.6 } }
.( verify {{0.1 0.2}{0.3 0.4}{0.5 0.6}} => ) .               

b 0 nn.b=                        \ set layer bias parameters
0 nn.b                           \ verify { 1 2 3 }
.( verify { 1 2 3 } => ) .

2 vector{ 10 20 }                \ model input tensor
1 1 2 1 reshape4
forward                          \ feed forward

-1 n@                            \ validate output layer { 6 13 20 }
.( verify { { 6 } { 13 } { 20 } } => ) .

bye
