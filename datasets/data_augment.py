# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/7/24 3:23
# @Function      : expand dataset
import os
import math
import shutil

import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn

# set device here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adversarial_examples(inp, eps, new_image):
    # this is it, this is the method of adversarial training
    inp.data = inp.data + ((eps / 255.0) * torch.sign(inp.grad.data))
    inp.grad.data.zero_()  # unnecessary

    # de-process image
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    adv = inp.data.cpu().numpy()[0].transpose(1, 2, 0)
    adv = ((adv * std) + mean) * 255.0
    adv = np.clip(adv[..., ::-1], 0, 255).astype(np.uint8)  # RGB to BGR
    # save images
    cv2.imwrite(new_image, adv, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


def add_adversarial(model, ori_dir, exp_dir, ori_num, add_num, image_size=96):
    while add_num > 0:
        iter_num = math.ceil(add_num / ori_num)
        for image_name in os.listdir(ori_dir):
            if add_num <= 0:
                return
            for eps in range(1, iter_num+1):
                if add_num <= 0:
                    return
                os.chdir(ori_dir)
                orig = cv2.imread(image_name)[..., ::-1]
                os.chdir(exp_dir)
                new_image_name = "%s_exp_%s.png" % (image_name.split(".")[0], eps)
                if os.path.exists(new_image_name):
                    continue
                orig = cv2.resize(orig, (image_size, image_size))
                img = orig.copy().astype(np.float32)
                img /= 255.0
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                img = (img - mean) / std
                img = img.transpose(2, 0, 1)
                inp = torch.from_numpy(img).to(device).float().unsqueeze(0).requires_grad_(True)
                out = model(inp)
                pred = np.argmax(out.data.cpu().numpy())
                criterion = nn.CrossEntropyLoss()
                loss = criterion(out, torch.tensor([pred], dtype=torch.float32).to(device).long())
                # compute gradients
                loss.backward()
                adversarial_examples(inp, eps, new_image_name)
                add_num -= 1


def expand_adversarial(model, ori_dir, exp_dir, max_num=50):
    for ori_char_dir, dirs, files in os.walk(ori_dir, topdown=False):
        if dirs:
            continue
        exp_char_dir = os.path.abspath(ori_char_dir.replace(ori_dir, exp_dir))
        if not os.path.exists(exp_char_dir):
            shutil.copytree(ori_char_dir, exp_char_dir)
        exp_len = len(os.listdir(exp_char_dir))
        if exp_len >= max_num:
            continue
        print("expand directory:", ori_char_dir, exp_char_dir)
        add_num = max_num - exp_len
        add_adversarial(model, ori_char_dir, exp_char_dir, len(os.listdir(ori_char_dir)), add_num)


if __name__ == "__main__":
    """
    replace with your own dataset path
    """
    ori_root_dir, exp_root_dir = os.path.abspath("ancient_3_ori"), os.path.abspath("ancient_3_exp")
    expand_adversarial(models.alexnet(pretrained=True).eval().to(device), ori_root_dir, exp_root_dir, 50)
