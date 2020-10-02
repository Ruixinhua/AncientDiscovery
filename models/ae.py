# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui & Zhang Gechuan
# @Time          : 2020/5/10 16:51
# @Function      : This is the class of auto encoder class


import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, input_shape=96):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=input_shape)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed, code
