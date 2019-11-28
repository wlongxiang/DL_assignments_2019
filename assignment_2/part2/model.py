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
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_num_hidden
        self.vocabulary_size = vocabulary_size
        self.lstm_layers = nn.LSTM(input_size=self.vocabulary_size,
                                   hidden_size=self.lstm_num_hidden,
                                   num_layers=self.lstm_num_layers)
        # this is the full connected layer to produce final output
        self.fc_layer = nn.Linear(in_features=self.lstm_num_hidden, out_features=self.vocabulary_size)

    def forward(self, x):
        # x is of shape (batch_size, seq_length)
        # Implementation here...
        one_hot_encoder = torch.eye(self.vocabulary_size)
        x_ohe = one_hot_encoder[x].to(self.device)
        # according to pytorch docs: input of shape (seq_length, batch_size, input_size)
        x_ohe = x_ohe.permute(1, 0, 2)
        lstm_output, _ = self.lstm_layers(x_ohe)
        fc_out = self.fc_layer(lstm_output)
        # permute output to be of shape (batch_size, input_size, seq_length)
        out = fc_out.permute(1, 2, 0)
        return out
