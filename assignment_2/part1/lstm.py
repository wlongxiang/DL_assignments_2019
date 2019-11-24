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
        self.wg = nn.Parameter(torch.Tensor(num_hidden, input_dim + num_hidden))
        self.wgx = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.wgh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))

        self.bg = nn.Parameter(torch.Tensor(num_hidden))

        self.wi = nn.Parameter(torch.Tensor(num_hidden, input_dim+num_hidden))
        self.wix = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.wih = nn.Parameter(torch.Tensor(num_hidden, num_hidden))

        self.bi = nn.Parameter(torch.Tensor(num_hidden))

        self.wf = nn.Parameter(torch.Tensor(num_hidden, input_dim + num_hidden))
        self.wfx = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.wfh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bf = nn.Parameter(torch.Tensor(num_hidden))

        self.wo = nn.Parameter(torch.Tensor(num_hidden, input_dim + num_hidden))
        self.wox = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        self.woh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bo = nn.Parameter(torch.Tensor(num_hidden))

        self.wp = nn.Parameter(torch.Tensor(num_classes, num_hidden))
        self.bp = nn.Parameter(torch.Tensor(num_classes))

        for weight in [self.wg, self.wi, self.wf, self.wo, self.wp]:
            nn.init.xavier_uniform_(weight)

        for bias in [self.bg, self.bi, self.bo, self.bp]:
            nn.init.zeros_(bias)

        nn.init.ones_(self.bf)

        self.register_buffer('h0', torch.zeros(1, num_hidden))
        self.register_buffer('c0', torch.zeros(1, num_hidden))

    def forward(self, x):
        # Implementation here ...
        _seg_length = x.shape[1]
        if _seg_length != self.seq_length:
            raise ValueError("sequence length is {}, but {} is expected".format(_seg_length, self.seq_length))
        batch_size = x.shape[0]
        h_prev = self.h0.expand(batch_size, -1)
        c_prev = self.c0

        for t in range(self.seq_length):
            x_t = x[:, t:t + 1]
            x_h = torch.cat((x_t, h_prev), dim=-1)

            g = torch.tanh(x_h.matmul(self.wg.t()) + self.bg)
            i = torch.sigmoid(x_h.matmul(self.wi.t()) + self.bi)
            f = torch.sigmoid(x_h.matmul(self.wf.t()) + self.bf)
            o = torch.sigmoid(x_h.matmul(self.wo.t()) + self.bo)

            c = g * i + c_prev * f
            h_prev = torch.tanh(c) * o
            c_prev = c

        p = h_prev.matmul(self.wp.t()) + self.bp

        return p