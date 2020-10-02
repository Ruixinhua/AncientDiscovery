# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui & Zhang Gechuan
# @Time          : 2020/4/28 1:43
# @Function      : The helper for loss calculation
import torch
import pandas as pd
import torch.nn as nn
from helper.mmd import MMDLoss
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD


def beta_vae_loss(recon_x, x, mu, log_var, loss_type='H', stop_iter=1000, num_iter=None, max_capacity=25., gamma=1000.,
                  beta=8):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    C_max = torch.tensor([max_capacity])
    C_stop_iter = stop_iter
    gamma = gamma
    if loss_type == 'H':
        loss = MSE + beta * KLD
        return loss
    elif loss_type == 'B':
        C_max = C_max.to(x.device)
        C = torch.clamp(C_max / C_stop_iter * num_iter, 0, C_max.data[0])
        loss = MSE + gamma * (KLD - C).abs()
        return loss


def mmd_err(code_vectors1, code_vectors2, kernel_type="linear"):
    return MMDLoss(kernel_type=kernel_type).forward(code_vectors1, code_vectors2)


def coral(source, target):
    d = source.data.shape[1]
    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)
    return loss


def cal_dis_err(code_vectors1, code_vectors2, labels=None, train=True, criterion="mmd"):
    if train:
        return get_dis_err(code_vectors1, code_vectors2, labels, criterion)
    else:
        with torch.no_grad():
            return get_dis_err(code_vectors1, code_vectors2, labels, criterion)


def get_dis_err(code_vectors1, code_vectors2, labels=None, criterion="mmd", mse=nn.MSELoss(reduction="sum")):
    if labels:
        dis_err = torch.tensor(0, dtype=torch.float)
        index = 0
        labels_df = pd.DataFrame({"index": [i for i in range(len(labels))], "labels": labels})
        for _, group in labels_df.groupby("labels"):
            code1, code2 = code_vectors1[group["index"].to_numpy(), :], code_vectors2[group["index"].to_numpy(), :]
            if criterion == "mean":
                dis_err = dis_err + mse(code1, code2) if index else mse(code1, code2)
            elif criterion == "coral":
                dis_err = dis_err + coral(code1, code2) if index else coral(code1, code2)
            else:
                kernel_type = "rbf" if criterion == "mmd-rbf" else "linear"
                dis_err = dis_err + mmd_err(code1, code2, kernel_type) if index else mmd_err(code1, code2, kernel_type)
            index += 1
    else:
        if criterion == "mean":
            dis_err = mse(code_vectors1, code_vectors2)
        else:
            kernel_type = "rbf" if criterion == "mmd-rbf" else "linear"
            dis_err = mmd_err(code_vectors1, code_vectors2, kernel_type)
    return dis_err

