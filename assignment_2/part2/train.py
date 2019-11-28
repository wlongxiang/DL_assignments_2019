# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import csv
import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel


################################################################################
def predict(dataset, seq_length, model, device, custom_init_seq=None, tau=0):
    """
    Run prediction for LSTM model.

    :param Dataset dataset: the pytorch dataset object as in TextDataset implementation.
    :param int seq_length: the desired total sequence length for the final output of prediction
    :param model nn.Module: a pytorch model inherit from nn.Module API
    :param str device: weather to run this on "cpu" or "CUDA:0"
    :param str custom_init_seq: initial sequqence of seeds, such as "sleeping beauty", default to None where one random chara
        cter from corpus will be selected as seed
    :param float tau: the random sampling temperature. Default to 0, corresponding to greedy sampling.
    :return str: a generated sentense
    """
    if custom_init_seq:
        init_sequence = []
        for char in custom_init_seq:
            char_idx = dataset._char_to_ix[char]
            init_sequence.append(char_idx)
        init_sequence = torch.tensor(init_sequence).reshape([1, len(custom_init_seq)]).to(device)
    else:
        # if the init sequence is not specified we choose an random one from the corpus
        char_idx = dataset._char_to_ix[np.random.choice(dataset._chars)]
        init_sequence = torch.tensor(char_idx).reshape([1, 1]).to(device)

    for i in range(seq_length - init_sequence.shape[1]):
        raw_pred_letter_last = model.forward(init_sequence)[:, :, -1]  # shape (1, vocab_size)
        if tau == 0:
            predcited_letter = raw_pred_letter_last.argmax(dim=1).reshape(1, -1)
        else:
            letter_prob = torch.softmax(raw_pred_letter_last / tau, dim=1)
            predcited_letter = torch.multinomial(letter_prob, 1).reshape([1, -1])
        init_sequence = torch.cat((init_sequence, predcited_letter), 1)
    sentence = dataset.convert_to_string(init_sequence.squeeze().numpy())
    return sentence


def calc_accuracy(predictions, targets):
    """
    Calculate the training/test accuracy. Note that predictions is a tensor from one hot encoded result,
    the max of dim of vocab_size is required to before comparing with targets which has the label for each time step.
    After max operation, we just need to get number of correct predictions divided by all sequences in all batch.

    :param Tensor predictions: tensor of shape (batch_size, vocab_size, seq_length)
    :param Tensor targets: tensor of shape (batch_size, seq_length)
    :return float: accuracy
    """
    batch_size, sqe_length = targets.shape
    # dimension in vocab_size dimension to squeeze the one hot encoded results
    y_pred = predictions.argmax(dim=1)
    accuracy = torch.sum(y_pred == targets).item() / (batch_size * sqe_length)
    return accuracy


def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file, seq_length=config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size,
                                seq_length=config.seq_length,
                                # note that voccab size is number of characters, not words
                                vocabulary_size=dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers, device=config.device).to(device)  # fixme
    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    # here we use the adam optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=0)  # fixme
    # init csv file
    for d in ["results", "checkpoints", "assets"]:
        if not os.path.exists(d):
            os.mkdir(d)
    cvs_file = 'results/textgen_tau_{}_inputlength_{}_hiddenunits_{}_lr_{}_batchsize_{}_{}.csv'.format(config.tau,
                                                                                                       config.seq_length,
                                                                                                       config.lstm_num_hidden,
                                                                                                       config.learning_rate,
                                                                                                       config.batch_size,
                                                                                                       int(time.time()))
    cols_data = ['step', 'train_loss', 'train_accuracy']
    with open(cvs_file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(cols_data)
    text_gen = 'results/result_{}.txt'.format(int(time.time()))
    step = 0
    accuracy = 0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        batch_loss = criterion(outputs, batch_targets)
        batch_loss.backward()
        optimizer.step()
        # clip the norm to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        loss = batch_loss.item()  # fixme
        accuracy = calc_accuracy(outputs, batch_targets)  # fixme

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % config.print_every == 0:
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

        if step > 1 and step % 5000 == 0:
            print("saving model at step {}, accu {}".format(step, accuracy))
            torch.save(model, './checkpoints/step_{}_acc_{}.model'.format(step, accuracy))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            for i in range(1):
                sentense = predict(dataset, config.seq_length, model, device, custom_init_seq="sleeping beauty is ",
                                   tau=config.tau)
                print(sentense)
                with open(text_gen, 'a') as fp:
                    fp.write('{}:{}\n'.format(int(step), sentense))
            # another way

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
    print("saving model at step {}, accu {}".format(step, accuracy))
    torch.save(model, './checkpoints/final_step_{}_acc_{}.model'.format(step, accuracy))
    print('Done training.')


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default="./assets/book_EN_grimms_fairy_tails.txt",
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=40, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--tau', type=int, default=0.5, help="apply sampling to the generation process")

    config = parser.parse_args()

    # Train the model
    train(config)
