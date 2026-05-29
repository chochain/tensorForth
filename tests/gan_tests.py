#
# GAN tests - simple linear regression Z = X @ A + B
#
import torch
from torch import nn
from d2l import torch as d2l

X = torch.normal(0, 1, (1000, 2))             # 1000 random points (randn)
A = torch.tensor([[1, 2], [-0.1, 0.5]])       # real data direction
B = torch.tensor([1, 2])                      # real data bias
Z = torch.matmul(X, A) + B                    # Corpus Z = X @ A + B

class nnShim(nn.Module):
    def __init__(self):
        super(nnShim, self).__init__()
        self.v = []
    def forward(self, x):
        self.v = x
        return x

net_G = nn.Sequential(
    nn.Linear(2, 2))
net_D = nn.Sequential(
    nnShim(),
    nn.Linear(2, 5),
    nn.LeakyReLU(0.2),
    nn.Linear(5, 3),
    nn.LeakyReLU(0.2),
    nn.Linear(3, 1),
    nn.Sigmoid())
#@save

for w in net_D.parameters():
    nn.init.normal_(w, 0, 0.02)              # was 0.02 => 0.707
for w in net_G.parameters():
    nn.init.normal_(w, 0, 0.02)              # was 0.02 => 0.707
opti_D = torch.optim.Adam(net_D.parameters(), lr=0.02)    # was 0.02 => 0.001
opti_G = torch.optim.Adam(net_G.parameters(), lr=0.002)   # was 0.002=> 0.001
#@save

def dump():
    print("D[0]=", net_D[0].v)                # dump from nnShim capture
    for n, p in net_G[0].named_parameters():
        print("g[0]", n, p.data)
        
def update_D(Zb, Xb, loss):
    """Update discriminator."""
    N       = Zb.shape[0]
    ones    = torch.ones((N,), device=Zb.device)
    zeros   = torch.zeros((N,), device=Zb.device)
    opti_D.zero_grad()
    
    # D train real as real
    real_z  = net_D(Zb)
    loss_r  = loss(real_z, ones.reshape(real_z.shape))

    # D train fake as fake
    # Do not need to compute gradient for `net_G`, so detach it (no backprop)
    fake_x  = net_G(Xb)
    fake_z  = net_D(fake_x.detach())
    loss_f  = loss(fake_z, zeros.reshape(fake_z.shape))
    
    (loss_r + loss_f).backward()
    opti_D.step()
    
    return [ loss_r, loss_f ]
#@save

def update_G(Xb, loss):
    """Update generator."""
    N      = Xb.shape[0]
    ones   = torch.ones((N,), device=Xb.device)
    opti_G.zero_grad()
    
    # We could reuse `fake_x` from `update_D` to save computation
    # Recomputing `fake_z` is needed since `net_D` is changed
    # G train fake as real 
    fake_x = net_G(Xb)
    fake_z = net_D(fake_x)
    loss_g = loss(fake_z, ones.reshape(fake_z.shape))
    loss_g.backward()
    opti_G.step()
    
    return loss_g

def train(batch_size, data_iter, num_epochs=20, latent_dim=2):
    #loss = nn.BCEWithLogitsLoss(reduction='sum')   # D without Sigmoid output
    loss = nn.BCELoss(reduction='mean')             # D with    Sigmoid output
    for epoch in range(num_epochs):
        # Train one epoch
        timer  = d2l.Timer()
        metric = d2l.Accumulator(4)                 # loss_r, loss_f, loss_g, N
        for (Zb,) in data_iter:                     # a batch of Z
            N  = Zb.shape[0]
            Xb = torch.normal(0, 1, size=(N, latent_dim))  # create a batch of X
            d  = update_D(Zb, Xb, loss)
            g  = update_G(Xb, loss)
            metric.add(d[0], d[1], g, N)
        # Show the losses
        nb  = metric[3]/batch_size                  # number of batches
        rps = nb/timer.stop()
        xr, xf, xg = metric[0]/nb, metric[1]/nb, metric[2]/nb
        print(f'====> G={xg:.3f}, Dr={xr:.3f}, Df={xf:.3f} ({rps:.1f} rps)')
        dump();
    
batch_size = 10
data_iter  = d2l.load_array((Z,), batch_size)
train(batch_size, data_iter)

print("net_G=", net_G[0].weight, net_G[0].bias)
fake_z = net_G(torch.normal(0, 1, size=(100, 2)))
print(fake_z)


