# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/9/10 17:17
# @Function      : Generic runner for SDS and PDS models
import argparse
import shutil

from helper import AncientDataset, ModelConfiguration
from trainers import SingleDecoderTrainer, PairedDecoderTrainer
from util import tools


def experiment(config_file, root_dir):
    """
    Normal experiment process

    Args:
        config_file: the path to configuration file
        root_dir: the root directory of dataset

    """
    # set configuration
    config = tools.load_config(config_file)
    conf = ModelConfiguration(**config)
    shutil.copy(conf_path, conf.log_path+"/"+"config.yaml")

    # set dataset
    dataset = AncientDataset(conf=conf, root_dir=root_dir)
    log_file = open(conf.log_file, "a+", encoding="utf-8")
    tools.print_log("Split Dataset", file=log_file)
    dataset.split_dataset(batch_size=32)

    # set trainer
    tools.print_log("Start Training", file=log_file)
    is_single = conf.strategy == "single"
    trainer = SingleDecoderTrainer(conf, dataset) if is_single else PairedDecoderTrainer(conf, dataset)
    trainer.train()


if __name__ == "__main__":
    # set argument
    parser = argparse.ArgumentParser(description="Generic runner for SDS and PDS structure")
    conf_path, root = "../configs/svq_jj.yaml", "../datasets/"
    parser.add_argument("--config", "-c", dest="filename", metavar="FILE", help="config file path",  default=conf_path)
    parser.add_argument("--root", "-r", dest="root", metavar="TEXT", help="dataset root directory", default=root)
    args = parser.parse_args()
    experiment(args.filename, args.root)
