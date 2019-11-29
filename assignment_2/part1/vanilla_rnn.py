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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.device = device
        self.input_dim = input_dim
        # input-to-hidden matrix
        self.U = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.bh = nn.Parameter(torch.Tensor(num_hidden))
        self.V = nn.Parameter(torch.Tensor(num_classes, num_hidden))
        self.bp = nn.Parameter(torch.Tensor(num_classes))
        self.W = nn.Parameter(torch.Tensor(num_hidden, num_hidden), requires_grad=True)
        # init
        # vanilla rnn seems to be sensitive to weights init, xavier normal seems to work well
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.bh)
        nn.init.zeros_(self.bp)
        self.hidden_states = []
        self.hidden_weights = []

    def forward(self, x):
        # Implementation here ...
        _seg_length = x.shape[1]
        if _seg_length != self.seq_length:
            raise ValueError("sequence length is {}, but {} is expected".format(_seg_length, self.seq_length))
        batch_size = x.shape[0]
        self.W.requires_grad_(True)
        hidden_state_prev_seq = torch.zeros_like(self.bh)
        hidden_state_prev_seq.requires_grad_(True)
        for t in range(self.seq_length):
            # x_ts represents the value at specific time step t, with length of batch size
            x_ts = x[:, t:t + self.input_dim].reshape(batch_size, self.input_dim)
            # apply a linear transform with hidden state and bias
            linear_transformed = x_ts @ self.U.data.T + hidden_state_prev_seq @ self.W.data.T + self.bh
            # apply tanh non-linearity
            hidden_state_prev_seq = torch.tanh(linear_transformed)
            self.W.retain_grad()
            self.hidden_weights.append(self.W)
            self.hidden_states.append(hidden_state_prev_seq)
            hidden_state_prev_seq.retain_grad()
            # apply output linear transform with bias
        output = hidden_state_prev_seq @ self.V.T + self.bp
        # p represents the ouput at the last time step here, note that the softmax part is done in the loss function
        # do not include it in the output layer
        return hidden_state_prev_seq, output
