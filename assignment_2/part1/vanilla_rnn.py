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

import math

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        # input-to-hidden matrix
        self.U = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.bh = nn.Parameter(torch.Tensor(num_hidden))
        self.V = nn.Parameter(torch.Tensor(num_classes, num_hidden))
        self.bp = nn.Parameter(torch.Tensor(num_classes))
        self.W = nn.Parameter(torch.Tensor(num_hidden, num_hidden), requires_grad=True)
        # init
        # it show great importance by experiment here that weights need to be
        # somehow have a small negtive mean, otherwise training just fails
        stdv = 1.0 / math.sqrt(num_hidden)
        for weight in [self.U, self.W, self.V]:
            nn.init.uniform_(weight, -stdv, stdv)
        nn.init.zeros_(self.bh)
        nn.init.zeros_(self.bp)
        self.hidden_init = torch.zeros_like(self.bh)

    def forward(self, x):
        # Implementation here ...
        _seg_length = x.shape[1]
        if _seg_length != self.seq_length:
            raise ValueError("sequence length is {}, but {} is expected".format(_seg_length, self.seq_length))
        batch_size = x.shape[0]

        h_prev = self.hidden_init
        for t in range(self.seq_length):
            # x_t represents the value at specific time step t, with length of batch size
            x_t = x[:, t:t + 1]
            h_prev = torch.tanh(x_t.matmul(self.U.t())
                                + h_prev.matmul(self.W.t())
                                + self.bh)

            p = h_prev.matmul(self.V.t()) + self.bp
        # p represents the ouput at the last time step here, note that the softmax part is done in the loss function
        # do not include it in the output layer
        return p
