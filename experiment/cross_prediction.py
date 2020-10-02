import argparse

import torch
import os
from util import tools
from helper import ModelConfiguration, AncientDataset, Prediction
from helper.ancient_dataset import get_dataset_by_path
import pandas as pd
import numpy as np


def cal_top_ns_acc(paths, ns):
    return np.array([sum([len(paths[p]) for p in paths.keys() if p < n]) for n in ns])


def save_accuracy(acc_df, t, m, fss, paths, ns, acc_dir="statistic/accuracy"):
    acc_path = os.path.join(acc_dir, "%s_%s_%s.csv" % (t.replace("->", "_"), m, fss))
    acc = cal_top_ns_acc(paths, ns) / len(paths_test) * 100
    line = [t, m, fss] + ["%.2f" % a for a in acc]
    acc_df = acc_df.append(pd.DataFrame([line], columns=columns), ignore_index=True)
    acc_df.to_csv(acc_path)
    return acc_df


if __name__ == "__main__":
    # add argument here
    parser = argparse.ArgumentParser(description='get prediction result after 10-fold cross validation')
    root, space_size = "../datasets/", 600  # space size should larger than 600, or you can choose other top-n value
    top_ns = [1, 10, 20, 50, 100, 200, 400, 600]
    parser.add_argument('--root', '-r', dest="root", metavar='TEXT', help='dataset root directory', default=root)
    parser.add_argument('--size', '-s', dest="size", metavar='TEXT', help='remaining space size', default=space_size)
    args = parser.parse_args()

    # initial statistic directory and statistic data structure
    accuracy_dir = "statistic/accuracy"
    columns = ["Task", "Model", "FSS"] + ["Top-" + str(n) for n in top_ns]
    accuracy_nor_df, accuracy_exp_df = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    # convert some terms
    convert_map = {"paired": "PDS", "single": "SDS", "jia": "O", "jin": "B", "chu": "C"}

    # create directory
    tools.make_dir(accuracy_dir)

    for root, dirs, files in os.walk("log", topdown=False):
        if "config.yaml" not in files:  # check if there is configuration file
            continue
        # set configuration
        config = tools.load_config(os.path.join(root, "config.yaml"))
        conf = ModelConfiguration(**config)

        # set checkpoint and load model
        if not os.path.exists(conf.best_model_path):  # check if the best model exist
            continue
        checkpoint = torch.load(conf.best_model_path, map_location=conf.device)
        model = tools.get_model_class(conf.core, **conf.model_params)
        model.load_state_dict(checkpoint["model"])
        model = model.to(conf.device)

        # set test dataset and get source data
        test_path = "cross_dataset/chars_%s_test.csv" % ("_".join(conf.paired_chars))
        dataset = AncientDataset(conf=conf, root_dir=args.root)
        test_char = pd.read_csv(test_path)["test"]
        target_test, labels_test, paths_test = get_dataset_by_path(dataset.target_dir, dataset.transform, test_char)
        dataset.get_source_data()

        # set some terms
        task = "%s->%s" % (convert_map[conf.paired_chars[0]], convert_map[conf.paired_chars[1]])
        core = convert_map[conf.strategy]
        fss_nor, fss_exp = len(dataset.char_list), len(dataset.exp_chars)

        # set prediction class and make prediction
        pred = Prediction(target_test, labels_test, dataset.source_data_full, dataset.source_labels_full, paths_test,
                          core=conf.core, set_type="test", size=args.size, model=model)

        # make prediction on normal size and save statistic data
        nor_result, nor_paths = pred.predict(pred.get_classifier(with_cluster=False))
        accuracy_nor_df = save_accuracy(accuracy_nor_df, task, core, fss_nor, nor_paths, top_ns)

        # make prediction on expansion size and save statistic data
        pred.set_source(dataset.source_data_exp, dataset.source_labels_exp, set_type="exp")
        exp_result, exp_paths = pred.predict(pred.get_classifier(with_cluster=False))
        accuracy_exp_df = save_accuracy(accuracy_exp_df, task, core, fss_exp, exp_paths, top_ns)
