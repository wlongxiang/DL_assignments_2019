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

import torch

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.lstm = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers)
        self.linear = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self.vocabulary_size = vocabulary_size
        one_hot_codes = torch.eye(self.vocabulary_size)
        self.register_buffer('one_hot_codes', one_hot_codes)

    def forward(self, x):
        # x is of shape (batch_size, seq_length)
        # Implementation here...
        x_one_hot = self.one_hot_codes[x]
        # according to pytorch docs: input of shape (seq_length, batch_size, input_size)
        x_one_hot= x_one_hot.permute(1,0,2)
        out, _ = self.lstm(x_one_hot)
        p = self.linear(out)
        # permute output to be of shape (batch_size, input_size, seq_length)
        p=p.permute(1,2,0)
        return p
