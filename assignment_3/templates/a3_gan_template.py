import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import datasets
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=784),
            nn.Tanh()
        )

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

    def forward(self, z):
        # Generate images from z
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=1),
            # nn.ReLU()
        )
        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)


def sample_generator(generator, n_samples, z=None):
    """Obtain samples from the generator. The returned tensor is on device and
    attached to the graph, so it has requires_grad=True """
    if z is None:
        z = torch.randn(n_samples, args.latent_dim).to(device)

    samples = generator(z)
    return samples


def save_samples(generator, fname):
    samples = sample_generator(generator, n_samples=25).detach().cpu()
    samples = samples.reshape(-1, 1, 28, 28) * 0.5 + 0.5

    grid = make_grid(samples, nrow=5)[0]
    plt.cla()
    plt.imshow(grid.numpy(), cmap='binary')
    plt.axis('off')
    img_path = os.path.join(os.path.dirname(__file__), 'ganresults', fname)
    plt.savefig(img_path)
    # os.remove(img_path)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    ones = torch.ones((args.batch_size, 1), dtype=torch.float32).to(device)
    zeros = torch.zeros((args.batch_size, 1), dtype=torch.float32).to(device)
    bce_loss = nn.BCEWithLogitsLoss()

    train_iters = 0
    avg_loss_d = 0
    avg_loss_g = 0
    log = 'epoch [{:d}/{:d}] batch [{:d}/{:d}] loss_d: {:.6f} loss_g: {:.6f}'
    n_epochs = args.n_epochs

    for epoch in range(1, n_epochs + 1):
        if epoch == 1 or epoch % args.save_epochs == 0:
            fname = 'samples_{:d}.png'.format(epoch)
            save_samples(generator, fname)

        for i, (imgs, _) in enumerate(dataloader):
            # Train Discriminator
            # -------------------
            imgs = imgs.reshape(args.batch_size, -1).to(device)
            samples = sample_generator(generator, args.batch_size).detach()
            optimizer_D.zero_grad()
            if imgs.shape[1] != 784:
                print("!!!! image shap is {}, not of 784!!".format(imgs.shape[1]))
                break
            pos_preds = discriminator(imgs)
            neg_preds = discriminator(samples)
            # One-sided label smoothing
            ones.uniform_(0.7, 1.2)
            loss_d = bce_loss(pos_preds, ones) + bce_loss(neg_preds, zeros)
            loss_d.backward()
            optimizer_D.step()
            # -------------------

            # Train Generator
            # -------------------
            samples = sample_generator(generator, args.batch_size)
            optimizer_G.zero_grad()
            neg_preds = discriminator(samples)
            loss_g = bce_loss(neg_preds, ones)
            loss_g.backward()
            optimizer_G.step()
            # -------------------

            train_iters += 1
            avg_loss_d += loss_d.item() / args.log_interval
            avg_loss_g += loss_g.item() / args.log_interval

            if train_iters % args.log_interval == 0:
                print(log.format(epoch + 1, n_epochs,
                                 i + 1, len(dataloader),
                                 avg_loss_d, avg_loss_g))
                avg_loss_d = 0
                avg_loss_g = 0


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = DataLoader(datasets.MNIST('./data/mnist', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))])),
                            batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='log every LOG_INTERVAL iterations')
    parser.add_argument('--save_epochs', type=int, default=20,
                        help='save samples every SAVE_EPOCHS epochs')
    args = parser.parse_args()

    main()
