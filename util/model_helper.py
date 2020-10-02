# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 1:48
# @Function      : Some useful function of run models
import pickle
import torch
import torch.nn as nn
from util import tools, loss_helper
import torch.nn.functional as F


def run_batch(model, batch, core="AE", train=True, device=tools.get_device(), mse_loss=nn.MSELoss(reduction="sum"),
              num_iter=None, input_source=False):
    """
    Put a batch of data into models
    Args:
        model: Model object
        batch: Batch data
        core: Type of models
        train: Boolean type, default is True
        mse_loss: loss function of models
        device: cuda device, default is 0
        input_source: True or False
        num_iter: the number of iteration which is running

    Returns: The code and loss value after run the batch

    """
    batch = batch.to(device)
    if train:
        model.train()
        return run_model(model, batch, core, mse_loss, num_iter=num_iter, input_source=input_source)
    else:
        model.eval()
        with torch.no_grad():
            return run_model(model, batch, core, mse_loss, num_iter=num_iter, input_source=input_source)


def run_model(model, batch, core="AE", mse_loss=nn.MSELoss(reduction="sum"), input_source=True, img_size=96,
              num_iter=None):
    """
    Run models with batch data
    Args:
        model: Model object
        batch: Batch data
        core: Type of model
        mse_loss: loss function of model
        img_size: the size of image
        input_source: True or False
        num_iter: the number of iteration which is running

    Returns: The code and loss value after run the batch

    """
    if core == "AE":
        batch = batch.view(-1, img_size * img_size)
        output, code = model(batch)
        loss = mse_loss(output, batch)
    elif core == "VQVAE":
        output, code, vq_loss = model(batch)
        loss = vq_loss + mse_loss(batch, output)
    elif core == "VanillaVAE2":
        if input_source:
            recon1, z1, mu1, log_var1 = model.reconstruct1(batch)
            g1, z = model.generate_source(batch)
            recon_loss = loss_helper.vae_loss(recon1, batch, mu1, log_var1) + F.mse_loss(g1, batch, reduction='sum')
            return z, recon_loss
        else:
            recon2, z2, mu2, log_var2 = model.reconstruct2(batch)
            g2, z = model.generate_target(batch)
            recon_loss = loss_helper.vae_loss(recon2, batch, mu2, log_var2) + F.mse_loss(g2, batch, reduction='sum')
            return z, recon_loss
    else:
        output, code, mu, logvar = model(batch)
        if num_iter is not None:
            loss = loss_helper.beta_vae_loss(output, batch, mu, logvar, loss_type='H', num_iter=num_iter)
        else:
            loss = loss_helper.vae_loss(output, batch, mu, logvar)
    return code, loss


def get_center(model, batch, model_type="AE", num_iter=None):
    batch = torch.tensor(batch)
    if len(batch.shape) < 4:
        batch = batch.unsqueeze(0)
    # get the center of this batch
    code, _ = run_batch(model, batch, core=model_type, train=False, num_iter=num_iter)
    return torch.mean(code.cpu().data, dim=0).reshape((1, -1)).numpy()[0]
