# original source https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py

import argparse
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

from wgan_models import Generator, Discriminator


import torch as pt

def parse_options():
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    ar = parser.parse_args()
    print(ar)

    img_shape = (ar.channels, ar.img_size, ar.img_size)


    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    return ar, img_shape, device


# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super(Generator, self).__init__()
#         self.img_shape = img_shape
#         self.latent_dim = latent_dim
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *block(latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.shape[0], *self.img_shape)
#         return img
#
#     def get_noise(self, batch_size, device):
#         return pt.randn(batch_size, self.latent_dim, device=device)
#
# class Discriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.shape[0], -1)
#         validity = self.model(img_flat)
#         return validity


def train(dataloader, device, dis_opt, gen_opt, dis, gen, ar, epoch, batches_done):
    for idx, (imgs, _) in enumerate(dataloader):

        # Configure input
        # real_imgs = Variable(imgs.type(Tensor))
        imgs = imgs.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        dis_opt.zero_grad()

        # Sample noise as generator input
        z = gen.get_noise(imgs.shape[0], device)

        # Generate a batch of images
        fake_imgs = gen(z).detach()
        # Adversarial loss
        # loss_D = -pt.mean(discriminator(real_imgs)) + pt.mean(discriminator(fake_imgs))
        loss_D = -pt.mean(dis(imgs)) + pt.mean(dis(fake_imgs))

        loss_D.backward()
        dis_opt.step()

        # Clip weights of discriminator
        for p in dis.parameters():
            p.data.clamp_(-ar.clip_value, ar.clip_value)

        # Train the generator every n_critic iterations
        if idx % ar.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            gen_opt.zero_grad()

            # Generate a batch of images
            gen_imgs = gen(z)
            # Adversarial loss
            loss_G = -pt.mean(dis(gen_imgs))

            loss_G.backward()
            gen_opt.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, ar.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % ar.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1
    return batches_done

def main():
    ar, img_shape, device = parse_options()
    # Initialize generator and discriminator
    gen = Generator(ar.latent_dim, img_shape).to(device)
    dis = Discriminator(img_shape).to(device)

    # Configure data loader
    os.makedirs("../data", exist_ok=True)
    dataloader = pt.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=ar.batch_size,
        shuffle=True,
    )

    # Optimizers
    gen_opt = pt.optim.RMSprop(gen.parameters(), lr=ar.lr)
    dis_opt = pt.optim.RMSprop(dis.parameters(), lr=ar.lr)

    # Tensor = pt.cuda.FloatTensor if cuda else pt.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(ar.n_epochs):
        batches_done = train(dataloader, device, dis_opt, gen_opt, dis, gen, ar, epoch, batches_done)


if __name__ == '__main__':
    main()
