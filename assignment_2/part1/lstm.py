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

import torch
import torch.nn as nn


################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.device = device
        self.wgx = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.wgh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bg = nn.Parameter(torch.Tensor(num_hidden))

        self.wix = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.wih = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bi = nn.Parameter(torch.Tensor(num_hidden))

        self.wfx = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.wfh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bf = nn.Parameter(torch.Tensor(num_hidden))

        self.wox = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.woh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bo = nn.Parameter(torch.Tensor(num_hidden))

        self.wp = nn.Parameter(torch.Tensor(num_classes, num_hidden))
        self.bp = nn.Parameter(torch.Tensor(num_classes))

        # let's try kaiming normal here
        for w in [self.wgx, self.wgh,
                  self.wix, self.wih,
                  self.wfx, self.wfh,
                  self.wox, self.woh,
                  self.wp]:
            nn.init.kaiming_normal_(w)
        # init biases to zeros
        for bias in [self.bg, self.bi, self.bf, self.bo, self.bp]:
            nn.init.zeros_(bias)

    def forward(self, x):
        """
        Forward pass LSTM.

        :param Tensor x: x input of shape (batch_size, seq_length)
        :return Tensor: output of shape  (batch_size, vocab_size), vocab_size is 10 here because of 10 digits in numbers
            each row contains 10 probability for each batch
        """
        # Implementation here ...
        # check seq length validity
        _seg_length = x.shape[1]
        if _seg_length != self.seq_length:
            raise ValueError("sequence length is {}, but {} is expected".format(_seg_length, self.seq_length))

        # init hidden state and cell state
        batch_size = x.shape[0]
        hidden_state_prev_seq = torch.zeros(size=(batch_size, self.num_hidden))
        cell_state_prev_seq = torch.zeros(size=(batch_size, self.num_hidden))

        for t in range(self.seq_length):
            x_ts = x[:, t:t + 1]
            g = torch.tanh(x_ts @ self.wgx.T + hidden_state_prev_seq @ self.wgh.T + self.bg)
            i = torch.sigmoid(x_ts @ self.wix.T + hidden_state_prev_seq @ self.wih.T + self.bi)
            f = torch.sigmoid(x_ts @ self.wfx.T + hidden_state_prev_seq @ self.wfh.T + self.bf)
            o = torch.sigmoid(x_ts @ self.wox.T + hidden_state_prev_seq @ self.woh.T + self.bo)
            cell_state_prev_seq = g * i + cell_state_prev_seq * f
            hidden_state_prev_seq = torch.tanh(cell_state_prev_seq) * o
        output = hidden_state_prev_seq @ self.wp.T + self.bp
        return hidden_state_prev_seq, output
