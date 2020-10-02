# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Zhang Gechuan
# @Time          : 2020/8/5 10:52
# @Function      : VanillaVAE With Paired Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VanillaVAE2(nn.Module):
    def __init__(self, in_channels=3, input_size=96, latent_dim=128, hidden_dims=5):
        super(VanillaVAE2, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = [2**i for i in range(5, hidden_dims+5)]
        self.decode_hidden_dims = [2**i for i in range(hidden_dims+4, 4, -1)]
        self.output_size = input_size

        # Build Encoder
        modules = []
        channel = in_channels
        for h_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()))
            channel = h_dim
            self.output_size = math.floor((self.output_size + 2 * 1 - 3) / 2 + 1)

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.output_size ** 2, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.output_size ** 2, self.latent_dim)

        # Build Decoder1
        modules1 = []
        self.decoder_input1 = nn.Linear(self.latent_dim, self.hidden_dims[-1] * self.output_size ** 2)
        for i in range(len(self.decode_hidden_dims) - 1):
            modules1.append(nn.Sequential(
                nn.ConvTranspose2d(self.decode_hidden_dims[i], self.decode_hidden_dims[i + 1], kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(self.decode_hidden_dims[i + 1]),
                nn.ReLU()))

        self.decoder1 = nn.Sequential(*modules1)
        self.final_layer1 = nn.Sequential(
            nn.ConvTranspose2d(self.decode_hidden_dims[-1], self.decode_hidden_dims[-1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(self.decode_hidden_dims[-1]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.decode_hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Tanh())

        # Build Decoder2
        modules2 = []
        self.decoder_input2 = nn.Linear(self.latent_dim, self.hidden_dims[-1] * self.output_size ** 2)
        for i in range(len(self.decode_hidden_dims) - 1):
            modules2.append(nn.Sequential(
                nn.ConvTranspose2d(self.decode_hidden_dims[i], self.decode_hidden_dims[i + 1], kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(self.decode_hidden_dims[i + 1]),
                nn.ReLU()))

        self.decoder2 = nn.Sequential(*modules2)
        self.final_layer2 = nn.Sequential(
            nn.ConvTranspose2d(self.decode_hidden_dims[-1], self.decode_hidden_dims[-1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(self.decode_hidden_dims[-1]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.decode_hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Tanh())

    def encode(self, image):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param image: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(image)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode1(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input1(z)
        result = result.view(-1, self.decode_hidden_dims[0], self.output_size, self.output_size)
        result = self.decoder1(result)
        result = self.final_layer1(result)
        result = F.interpolate(result, size=(96, 96), mode='bilinear', align_corners=True)
        return result

    def decode2(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input2(z)
        result = result.view(-1, self.decode_hidden_dims[0], self.output_size, self.output_size)
        result = self.decoder2(result)
        result = self.final_layer2(result)
        result = F.interpolate(result, size=(96, 96), mode='bilinear', align_corners=True)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def reconstruct1(self, image):
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        return self.decode1(z), z, mu, log_var

    def reconstruct2(self, image):
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        return self.decode2(z), z, mu, log_var

    def generate_source(self, image1):
        with torch.no_grad():
            mu1, log_var1 = self.encode(image1)
            z1 = self.reparameterize(mu1, log_var1)
            recon2 = self.decode2(z1)
        mu2, log_var2 = self.encode(recon2)
        z2 = self.reparameterize(mu2, log_var2)
        recon1 = self.decode1(z2)
        return recon1, z2

    def generate_target(self, image2):
        with torch.no_grad():
            mu2, log_var2 = self.encode(image2)
            z2 = self.reparameterize(mu2, log_var2)
            recon1 = self.decode1(z2)
        mu1, log_var1 = self.encode(recon1)
        z1 = self.reparameterize(mu1, log_var1)
        recon2 = self.decode2(z1)
        return recon2, z2
