################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import time
from datetime import datetime
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################
def get_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = targets.shape[0]
    _, y_pred = predictions.max(dim=1)
    accuracy = (y_pred == targets).sum().item() / batch_size
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model_def = VanillaRNN
    else:
        model_def = LSTM
    model = model_def(config.input_length, config.input_dim,
                        config.num_hidden, config.num_classes,
                        config.device).to(device) # fixme
    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(), config.learning_rate)  # fixme
    # init csv file
    cvs_file = 'results/train{}_inputlength_{}_hiddenunits_{}_lr_{}_batchsize_{}_{}.csv'.format(config.model_type,
                                                                                                config.input_length,
                                                                                                config.num_hidden,
                                                                                                 config.learning_rate,
                                                                                                 config.batch_size,
                                                                                                 int(time.time()))
    cols_data = ['step', 'train_loss', 'train_accuracy']
    with open(cvs_file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(cols_data)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        out = model.forward(batch_inputs)
        batch_loss = criterion(out, batch_targets)
        batch_loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...
        optimizer.step()

        loss = batch_loss.item() # fixme
        accuracy = get_accuracy(out, batch_targets) # fixme

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            csv_data = [step, loss, accuracy]
            with open(cvs_file, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow(csv_data)
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=40, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)