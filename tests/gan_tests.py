import torch
from torch import nn
from d2l import torch as d2l

X = torch.normal(0.0, 1, (1000, 2))
A = torch.tensor([[1, 2], [-0.1, 0.5]])
b = torch.tensor([1, 2])
data = torch.matmul(X, A) + b

batch_size = 8
data_iter  = d2l.load_array((data,), batch_size)
net_G = nn.Sequential(nn.Linear(2, 2))
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.LeakyReLU(0.2),
    nn.Linear(5, 3), nn.LeakyReLU(0.2),
    nn.Linear(3, 1), nn.Sigmoid())
#@save

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones   = torch.ones((batch_size,), device=X.device)
    zeros  = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_Dr = loss(real_Y, ones.reshape(real_Y.shape))
    loss_Df = loss(fake_Y, zeros.reshape(fake_Y.shape))
    loss_Dr.backward()
    loss_Df.backward()
    trainer_D.step()
    return [ loss_Dr, loss_Df ]
#@save

def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G

def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    #loss = nn.BCEWithLogitsLoss(reduction='sum')
    loss = nn.BCELoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(4)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            d = update_D(X, Z, net_D, net_G, loss, trainer_D)
            g = update_G(Z, net_D, net_G, loss, trainer_G)
            metric.add(d[0], d[1], g, batch_size)
        # Show the losses
        N   = metric[3]
        rps = N/timer.stop()
        xdr, xdf, xg = metric[0]/N, metric[1]/N, metric[2]/N
        print(f'Dr={xdr:.3f}, Df={xdf:.3f}, G={xg:.3f} {rps:.1f} examples/sec')
    
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, data[:100].detach().numpy())
