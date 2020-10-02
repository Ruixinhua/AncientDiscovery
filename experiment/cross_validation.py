# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/7/22 12:00
# @Function      : This is the training process for 10-fold cross validation
import argparse
import os
import shutil

import yaml

from util import tools
from helper import AncientDataset, ModelConfiguration
from trainers import SingleDecoderTrainer, PairedDecoderTrainer
import pandas as pd
import numpy as np


def create_cross_dataset(num, char_list, path_test, path_val):
    np.random.shuffle(char_list)
    split_num = round(len(char_list) / num)
    char_split = [char_list[i * split_num:(i + 1) * split_num] for i in range(num)]
    test_chars = char_split[0]
    char_split.remove(char_split[0])
    pd.DataFrame({"test": test_chars}, columns=["test"]).to_csv(path_test)

    def split_char(n):
        train = [c for c in char_list if c not in set(char_split[n])]
        return {"val": ",".join(char_split[n]), "train": ",".join(train)}

    combinations = pd.DataFrame([split_char(i) for i in range(len(char_split))])
    combinations["checked"] = 0
    combinations.to_csv(path_val)


if __name__ == "__main__":
    # cross validation with 10-fold
    test_num = 10
    test_nums = [i for i in range(0, test_num - 1)]
    parser = argparse.ArgumentParser(description='training process for 10-fold Cross validation')
    root = "../datasets/"
    parser.add_argument('--root', '-r', dest="root", metavar='TEXT', help='dataset root directory', default=root)
    args = parser.parse_args()
    for conf_path in os.scandir("../configs/"):
        # load configuration
        config = tools.load_config(conf_path.path)
        saved_path = os.path.join("_".join(config["paired_chars"]), config["core"]+"_"+config["level"])
        conf = ModelConfiguration(**config, saved_path=saved_path)

        tools.print_log("Task: " + "->".join(conf.paired_chars) + ", Core: " + conf.core)

        # load dataset, split train and validation dataset
        dataset = AncientDataset(conf=conf, root_dir=args.root)
        test_path = "cross_dataset/chars_%s_test.csv" % ("_".join(conf.paired_chars))
        val_path = "cross_dataset/chars_%s_val.csv" % ("_".join(conf.paired_chars))
        if not os.path.exists(val_path):
            tools.make_dir("cross_dataset")
            create_cross_dataset(test_num, dataset.char_list, test_path, val_path)

        char_lists_combination = pd.read_csv(val_path)
        i = 0
        for val_chars, train_chars in zip(char_lists_combination["val"], char_lists_combination["train"]):
            if i not in test_nums:
                # skip this time, if it has been evaluated
                i += 1
                continue
            tools.print_log("Training number %s" % i)
            conf.set_path(os.path.join(saved_path, "cross", str(i)))
            config["saved_path"] = os.path.join(saved_path, "cross", str(i))
            with open(conf.log_path + "/" + "config.yaml", "w") as w:
                yaml.safe_dump(config, w)
            dataset.split_dataset(train_chars=train_chars.split(","), val_chars=val_chars.split(","))

            # load trainer
            is_single = conf.strategy == "single"
            trainer = SingleDecoderTrainer(conf, dataset) if is_single else PairedDecoderTrainer(conf, dataset)
            # if os.path.exists(conf.best_model_path):
            #     trainer.resume_checkpoint(conf.best_model_path)
            #     trainer.config = conf
            trainer.train()
            char_lists_combination.loc[i, 'checked'] += 1
            i += 1

