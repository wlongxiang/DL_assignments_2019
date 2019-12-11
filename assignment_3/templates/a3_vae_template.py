import argparse
import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist
from scipy import stats
import numpy as np

IN_FEATURES = 28 * 28


class Encoder(nn.Module):
    """
    In the original Kingma paper, MLPs with Gaussian outputs are used as probalistic encoders (C.2). More specifically,
    Firstly inputs are passed through a linear layer then a tanh non-linearity
    Then a linear layer is used for calculate the mu of gaussian output,
    And another linear layer is used for generating log variances.
    """

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.linear_hidden_layer = nn.Linear(in_features=IN_FEATURES, out_features=hidden_dim)
        self.nonlinear_hidden_layer = torch.tanh
        self.linear_mu_layer = nn.Linear(in_features=hidden_dim, out_features=z_dim)
        self.linear_var_layer = nn.Linear(in_features=hidden_dim, out_features=z_dim)
        self.log_variance = None

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hidden_output = self.linear_hidden_layer(input)
        hidden_output = self.nonlinear_hidden_layer(hidden_output)

        mean = self.linear_mu_layer(hidden_output)
        self.log_variance = self.linear_var_layer(hidden_output)
        std = torch.sqrt(torch.exp(self.log_variance))
        return mean, std


class Decoder(nn.Module):
    """
In the original Kingma paper, MLPs with Bernoulli outputs are used as probalistic decoder (C.2). More specifically,
Firstly inputs are passed through a linear layer then a tanh non-linearity
Then a linear layer is used for calculate the mean of Bernoulli output, and the output is suqashed to 0 and 1 by sigmoid
"""

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.linear_hidden_layer = nn.Linear(in_features=z_dim, out_features=hidden_dim)
        self.nonlinear_hidden_layer = torch.tanh
        self.linear_mu_layer = nn.Linear(in_features=hidden_dim, out_features=IN_FEATURES)
        self.sigmoid = torch.sigmoid

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        hidden_output = self.linear_hidden_layer(input)
        hidden_output = self.nonlinear_hidden_layer(hidden_output)

        mean = self.linear_mu_layer(hidden_output)
        mean = self.sigmoid(mean)
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None
        mean, std = self.encoder(input)
        # now we have the mean and std of the gaussian, we can perform a sampling from it using the reparametric trick
        samples_z = mean + torch.rand_like(mean) * std
        # now we pass the samples to decoder to get the reconstructed x_hat
        x_hat = self.decoder(samples_z)
        # get BCE for binary cross entropy
        reconstruction_loss = nn.BCELoss(reduction='none').forward(x_hat, input).sum(dim=-1)
        regularization_loss = 0.5 * (torch.pow(std, 2) + mean ** 2 - self.encoder.log_variance - 1).sum(dim=-1)
        average_negative_elbo = reconstruction_loss.mean() + regularization_loss.mean()
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None
        z_init = torch.randn((n_samples, self.z_dim))
        im_means = self.decoder.forward(z_init)
        sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    idx = 0
    for idx, batch_img in enumerate(data):
        # batch_img = batch_img.to(device)
        avg_elbo_batch = model.forward(batch_img.view(-1, IN_FEATURES))

        average_epoch_elbo += avg_elbo_batch

        if model.training:
            optimizer.zero_grad()
            avg_elbo_batch.backward()
            optimizer.step()
    average_epoch_elbo = average_epoch_elbo / idx
    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def plot_sampling_results(model, filename, num_sampes=8):
    results_path = os.path.join(os.path.dirname(__file__), 'vaeresults')
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    samples, means = model.sample(num_sampes)
    samples = samples.reshape(-1, 1, 28, 28)
    means = means.reshape(-1, 1, 28, 28)
    arrays = make_grid(samples, nrow=2)[0]
    img_fname = 'sample_' + filename
    plt.imsave(os.path.join(results_path, img_fname), arrays.detach().numpy(), cmap="binary")
    print("saving img:", img_fname)

    arrays = make_grid(means, nrow=2)[0]
    img_fname = 'mean_' + filename
    plt.imsave(os.path.join(results_path, img_fname), arrays.detach().numpy(), cmap="binary")
    print("saving img:", img_fname)

def plot_manifold_2d(model, file_name):
    n_rows = 8
    xy_grids = stats.norm.ppf(np.linspace(start=0.02, stop=0.98, num=n_rows))
    xx, yy = np.meshgrid(xy_grids, xy_grids)
    z = torch.tensor(np.column_stack((xx.reshape(-1), yy.reshape(-1))), dtype=torch.float)
    # create inital value for z, then decode it / generate it
    x = model.decoder(z)
    x = x.reshape(-1, 1, 28, 28)
    # plotting
    grid = make_grid(x, nrow=n_rows, padding=0)[0]
    plt.cla()
    plt.imshow(grid.detach().numpy(), cmap='binary')
    plt.axis('off')
    print("saving: ", file_name)
    img_path = os.path.join(os.path.dirname(__file__), 'vaeresults', file_name)
    plt.savefig(img_path)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        # the random inputs before any training
        if epoch == 0:
            plot_sampling_results(model, "vae_gen_0_{}.png".format(int(time.time())))

        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch == 0 or epoch == ARGS.epochs - 1 or epoch == ARGS.epochs // 2 or epoch == ARGS.epochs // 3:
            print("epoch: ", epoch)
            print("saving samples...")
            file_name = "vae_gen_{}_{}.png".format(epoch + 1, int(time.time()))
            plot_sampling_results(model, file_name)
            file_name = "manifold_{}_{}.png".format(epoch + 1, int(time.time()))

        if ARGS.zdim == 2:
            plot_manifold_2d(model, file_name)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo_{}.pdf'.format(int(time.time())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
