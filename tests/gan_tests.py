#
# GAN tests - simple linear regression Y = a*X + b
import torch
from torch import nn
from d2l import torch as d2l

X = torch.normal(0, 1, (1000, 2))             # 1000 random points
A = torch.tensor([[1, 2], [-0.1, 0.5]])       # real data direction
b = torch.tensor([1, 2])                      # real data bias
Y = torch.matmul(X, A) + b

class Ldump(nn.Module):
    def __init__(self):
        super(Ldump, self).__init__()
    def forward(self, x):
        print(x)
        return x

net_G = nn.Sequential(nn.Linear(2, 2))
net_D = nn.Sequential(
    nn.Linear(2, 5), Ldump(), nn.LeakyReLU(0.2),
    nn.Linear(5, 3), nn.LeakyReLU(0.2),
    nn.Linear(3, 1), nn.Sigmoid())
#@save

for w in net_D.parameters():
    nn.init.normal_(w, 0, 0.707)   # was 0.02
for w in net_G.parameters():
    nn.init.normal_(w, 0, 0.707)   # was 0.02
trainer_D = torch.optim.Adam(net_D.parameters(), lr=0.01)   # was 0.02
trainer_G = torch.optim.Adam(net_G.parameters(), lr=0.001)  # was 0.002
#@save

def dump_D(hdr):
    for n, p in net_D[4].named_parameters():
        print(hdr, n, p.data)
        
def update_D(X, Z, loss):
    """Update discriminator."""
    N       = X.shape[0]
    ones    = torch.ones((N,), device=X.device)
    zeros   = torch.zeros((N,), device=X.device)
    trainer_D.zero_grad()
    real_Y  = net_D(X)
    loss_Dr = loss(real_Y, ones.reshape(real_Y.shape))
    dump_D("<<<")
    print(f"loss={loss_Dr:.3f}")
    loss_Dr.backward()
    trainer_D.step()
    dump_D(">>>")
    return [ loss_Dr, 0 ]
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_X  = net_G(Z)
    fake_Y  = net_D(fake_X.detach())
    loss_Df = 0 #loss(fake_Y, zeros.reshape(fake_Y.shape))
    #loss_Df.backward()
    print(fake_Y, f"loss={loss_Dr:.3f}")
    trainer_D.step()
    return [ loss_Dr, loss_Df ]
#@save

def update_G(Z, loss):
    """Update generator."""
    return 0
    N      = Z.shape[0]
    ones   = torch.ones((N,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G

def train(bsz, data_iter, num_epochs=50, latent_dim=2):
    #loss = nn.BCEWithLogitsLoss(reduction='sum')   # D without Sigmoid output
    loss = nn.BCELoss(reduction='mean')             # D with    Sigmoid output
    for epoch in range(num_epochs):
        # Train one epoch
        timer  = d2l.Timer()
        metric = d2l.Accumulator(4)                 # loss_Dr, loss_Df, loss_G, N
        x = 0
        for (X,) in data_iter:
            n = X.shape[0]
            Z = torch.normal(0, 1, size=(n, latent_dim))
            d = update_D(X, Z, loss)
            g = update_G(Z, loss)
            metric.add(d[0], d[1], g, n)
            x += 1
            if x>5: break
        # Show the losses
        N   = metric[3]/bsz
        rps = N/timer.stop()
        xdr, xdf, xg = metric[0]/N, metric[1]/N, metric[2]/N
        print(f'====> Dr={xdr:.3f}, Df={xdf:.3f}, G={xg:.3f} ({rps:.1f} rps)')
        break
    
batch_size = 10
data_iter = d2l.load_array((Y,), batch_size)
train(batch_size, data_iter)
print("net_G=", net_G[0].weight, net_G[0].bias)


