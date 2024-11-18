import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.style.use('ggplot')

BATCH_SZ  = 100  # mini-batch size (600 batches/epoch)
EPOCHS    = 100  # total epoch
LAT_SZ    = 128  # latent vector size
GRID_SZ   = 64   # final 8x8 grid image
K         = 1    # number of steps to apply to the discriminator

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])
to_pil     = transforms.ToPILImage()
train_data = datasets.MNIST(
    root      = '../data',
    train     = True,
    download  = True,
    transform = transform
)
train_loader = DataLoader(train_data, batch_size=BATCH_SZ, shuffle=True)
img, lbl = next(iter(train_loader))
print(img.shape);

for i in range(100):
    ax = plt.subplot(10,10,i+1)
    #ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #plt.title(lbl[i])
    plt.imshow(img[i][0], cmap='gray_r')
plt.axis('off')
plt.show()

class nnShim(nn.Module):
    def __init__(self):
        super(nnShim, self).__init__()
        self.f = []
    def forward(self, x):
        self.f = x
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.sz   = LAT_SZ
        self.main = nn.Sequential(
            nn.Linear(self.sz, 256),
            nn.LeakyReLU(0.2),
#            nn.Linear(256, 512),
#            nn.LeakyReLU(0.2),
#            nn.Linear(512, 1024),
#            nn.LeakyReLU(0.2),
#            nn.Linear(1024, 784),
            nn.Linear(256, 784),
            nn.Tanh(),             # =>range between -1 and 1
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784        # 28x28
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
#            nn.Linear(self.n_input, 1024),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.3),
#            nn.Linear(1024, 512),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.3),
#            nn.Linear(512, 256),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)

generator     = Generator().to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('\n##### DISCRIMINATOR #####')
print(discriminator)

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# losses
criterion = nn.BCELoss()
losses_g  = [] # to store generator loss after each epoch
losses_dr = [] # to store discriminator loss after each epoch
losses_df = [] # to store discriminator loss after each epoch
images    = [] # to store images generatd by the generator

# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)
# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)

# function to create the noise vector
def create_noise(size):
    return torch.randn(size, LAT_SZ).to(device)

# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    N           = data_real.size(0)
    real_label  = label_real(N)
    fake_label  = label_fake(N)
    
    optimizer.zero_grad()
    
    output_real = discriminator(data_real)
    loss_real   = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake   = criterion(output_fake, fake_label)
    
    loss_real.backward()
    loss_fake.backward()
    
    optimizer.step()
    return [ loss_real, loss_fake ]

# function to train the generator network
def train_generator(optimizer, data_fake):
    N          = data_fake.size(0)
    real_label = label_real(N)
    
    optimizer.zero_grad()
    
    output = discriminator(data_fake)
    loss   = criterion(output, real_label)
    
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(EPOCHS):
    loss_g  = 0.0
    loss_dr = 0.0
    loss_df = 0.0
    BCNT    = int(len(train_data)/train_loader.batch_size)
    for bi, data in tqdm(enumerate(train_loader), total=BCNT):
        image, _ = data
        image    = image.to(device)
        N        = len(image)
        # run the discriminator for k number of steps
        for step in range(K):
            data_real = image
            data_fake = generator(create_noise(N)).detach()
            #
            # CC: detach breaks tensor backprop chain
            #     so that discriminator stop fake backprop into generator
            #     Keras, instead, uses Model.trainable = false
            #
            loss_d = train_discriminator(optim_d, data_real, data_fake)
            loss_dr += loss_d[0]
            loss_df += loss_d[1]
            
        data_fake = generator(create_noise(N))
        # train the generator network
        loss_g += train_generator(optim_g, data_fake)
        
    # create the final image (on CPU) for the epoch
    generated_img = generator(create_noise(GRID_SZ)).cpu().detach()
    # make the images as grid
    generated_img = make_grid(generated_img)
    
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"../out/gen_img{epoch}.png")
    images.append(generated_img)
    
    epoch_loss_g  = loss_g  / (bi+1) # total generator loss for the epoch
    epoch_loss_dr = loss_dr / (bi+1) # total discriminator loss for the epoch
    epoch_loss_df = loss_df / (bi+1) # total discriminator loss for the epoch
    
    losses_g.append(epoch_loss_g)
    losses_dr.append(epoch_loss_dr)
    losses_df.append(epoch_loss_df)

    print(f"Epoch {epoch} of {EPOCHS}")
    print(f"Loss G: {epoch_loss_g:.4f}, Dr: {epoch_loss_dr:.4f}, Df: {epoch_loss_df:.4f}")

print('DONE TRAINING')
torch.save(generator.state_dict(), '../out/generator.pth')

# save the generated images as GIF file
imgs = [np.array(to_pil(img)) for img in images]
imageio.mimsave('../out/generator_images.gif', imgs)

# plot and save the generator and discriminator loss
plt.figure()
plt.plot(torch.tensor(losses_g, device='cpu'),  label='Loss G')
plt.plot(torch.tensor(losses_dr, device='cpu'), label='Loss Dr')
plt.plot(torch.tensor(losses_df, device='cpu'), label='Loss Df')
plt.legend()
plt.savefig('../out/loss.png')


