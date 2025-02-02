import argparse
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def log_prior(x):
    """
    x is a Tensor of shape (n,)
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    # using the formular for standard normal distribution here
    # logp = -torch.log(2 * np.pi * torch.exp(torch.Tensor(x ** 2))) / 2.0
    logp = -0.5 * np.log(2 * np.pi) - 0.5 * x ** 2
    return logp.sum(-1)


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    # we can use the torch.rand for sampling from a standard normal distribution
    sample = torch.randn(size=size)
    if torch.cuda.is_available():
        sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28 * 28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # translation is mean, scale is like sigma li gaussian
        # Suggestion: Linear ReLU Linear ReLU Linear.
        # c_in and c_out stays the same dimensions
        self.nn = torch.nn.Sequential(
            nn.Linear(in_features=c_in, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=2 * c_in),
        )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        """
        Read this blog:
        https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#linear-algebra-basics-recap
        """
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.
        #  note that this nn represents the s and t function in the post mentioned below
        z_passed = self.nn(z * self.mask)
        # let's chunk z into two halves as described in here:
        # https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#linear-algebra-basics-recap
        s, t = torch.chunk(z_passed, chunks=2, dim=1)
        # t = torch.tanh(t)
        if not reverse:
            # note that mask is binary, we can do 1-mask to get the other part
            z_out = z * self.mask + (1 - self.mask) * (t + z * torch.exp(torch.tanh(s)))
            ldj = ldj + torch.sum((1 - self.mask) * torch.tanh(s), dim=1)
        else:
            # note that mask is binary, we can do 1-mask to get the other part
            z_out = z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-torch.tanh(s)))
            ldj = ldj - torch.sum((1 - self.mask) * torch.tanh(s), dim=1)
        z = z_out
        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1 - mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z * (1 - alpha) + alpha * 0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1 - z), dim=1)
            z = torch.log(z) - torch.log(1 - z)

        else:
            # Inverse normalize
            z = torch.sigmoid(z)
            logdet += torch.sum(torch.log(z) + torch.log(1 - z), dim=1)
            z = (z - alpha * 0.5) / (1 - alpha)

            # Multiply by 256.
            logdet += np.log(256) * np.prod(z.size()[1:])
            z = z * 256.

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_pz = log_prior(z)
        #
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)
        z, ldj = self.flow(z, ldj, reverse=True)
        z, ldj = self.logit_normalize(z, ldj, reverse=True)
        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    all_bpd = []
    for i, (imgs, _) in enumerate(data):
        imgs = imgs.reshape(-1, 784).to(device)
        neg_log_likelihood = - model(imgs).mean().to(device)
        if model.training:
            optimizer.zero_grad()
            neg_log_likelihood.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        all_bpd.append(neg_log_likelihood.item() / np.log(2))

    return np.mean(all_bpd) / 784


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------
        if epoch % 2 == 0 or epoch == ARGS.epochs - 1:
            samples = model.sample(25)
            samples = samples.view(-1, 1, 28, 28)
            arrays = make_grid(samples, nrow=5,normalize=True)[0]
            img_path = os.path.join(os.path.dirname(__file__), 'images_nfs',
                                    "sample_{}_{}.png".format(epoch, int(time.time())))
            print("saving image", img_path)

            #             save_image(samples, img_path,nrow=5,normalize=True)
            if arrays.is_cuda:
                arrays = arrays.cpu()
            plt.imsave(img_path, arrays.detach().numpy(), cmap="binary")

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()
    main()
