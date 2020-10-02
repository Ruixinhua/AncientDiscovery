# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/7/25 20:18
# @Function      : prediction class
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from util import cluster, model_helper


class Prediction:

    def __init__(self, target_data, target_labels, source_data, source_labels, paths, size=10, core="AE",
                 mode="instance", batch_size=256, set_type="val", model=None):
        self.target_data, self.target_labels, self.paths = target_data, target_labels, paths
        self.source_data, self.source_labels = source_data, source_labels
        self.batch_size, self.size, self.core, self.mode = batch_size, size, core, mode
        self.set_type, self.model = set_type, model

    def __get_output_df(self, model, data, labels, i, input_source=True):
        data = torch.tensor(data[i * self.batch_size:(i + 1) * self.batch_size])
        output, _ = model_helper.run_batch(model, data, self.core, False, input_source=input_source)
        output = [o.reshape(1, -1).squeeze(0) for o in output.cpu().numpy()]
        labels = labels[i * self.batch_size:(i + 1) * self.batch_size]
        return {"feature": output, "label": labels}

    def set_source(self, source_data, source_labels, set_type=None, mode=None):
        self.source_data, self.source_labels = source_data, source_labels
        self.set_type = set_type if set_type else self.set_type
        self.mode = mode if mode else self.mode

    def set_target(self, target_data, target_labels, paths, set_type=None, mode=None):
        self.target_data, self.target_labels, self.paths = target_data, target_labels, paths
        self.set_type = set_type if set_type else self.set_type
        self.mode = mode if mode else self.mode

    @staticmethod
    def _get_source_cluster(source_outputs, with_cluster):
        source_centers, source_labels_mapping = [], []
        if with_cluster:
            source_outputs = cluster.get_cluster_output_df(source_outputs, add_center=False)
            for (label, center), group_df in source_outputs.groupby(["label", "center"]):
                output = [o for o in group_df["feature"]]
                source_centers.append(np.mean(output, axis=0))
                source_labels_mapping.append(label+str(center))
        else:
            for label, group_df in source_outputs.groupby(["label"]):
                output = [o for o in group_df["feature"]]
                source_centers.append(np.mean(output, axis=0))
                source_labels_mapping.append(label)
        return source_centers, source_labels_mapping

    def get_source_output(self, source_paths=None):
        source_outputs = pd.DataFrame({"feature": [], "label": [], "type": []})
        for i in range(math.ceil(len(self.source_data) / self.batch_size)):
            output = self.__get_output_df(self.model, self.source_data, self.source_labels, i, True)
            if source_paths:
                paths = source_paths[i * self.batch_size:(i + 1) * self.batch_size]
                output["path"] = paths
            output.update({"type": ["source" for _ in range(len(output["feature"]))]})
            source_outputs = source_outputs.append(pd.DataFrame(output), ignore_index=True)
        return source_outputs

    def _get_source(self, source_outputs, with_cluster=False):
        if "path" in source_outputs:
            source_centers, source_labels_mapping = [], []
            for label, group_df in source_outputs.groupby(["label"]):
                output = [o for o in group_df["feature"]]
                source_centers.extend(output)
                source_labels_mapping.extend([p for p in group_df["path"]])
        else:
            source_centers, source_labels_mapping = self._get_source_cluster(source_outputs, with_cluster)
        self.source_labels_mapping = source_labels_mapping
        return source_centers, source_labels_mapping

    def get_classifier(self, with_cluster=False, source_outputs=pd.DataFrame(), source_paths=None):
        if source_outputs.empty:
            source_outputs = self.get_source_output(source_paths)
        source_centers, source_labels_mapping = self._get_source(source_outputs, with_cluster)
        classifier = KNeighborsClassifier(n_neighbors=self.size)
        classifier.fit(source_centers, source_labels_mapping)
        return classifier

    def set_model(self, model):
        self.model = model

    def get_target(self):
        target_outputs = pd.DataFrame({"feature": [], "label": [], "path": []})
        for i in range(math.ceil(len(self.target_data) / self.batch_size)):
            output = self.__get_output_df(self.model, self.target_data, self.target_labels, i, False)
            paths = self.paths[i * self.batch_size:(i + 1) * self.batch_size]
            output["path"] = paths
            target_outputs = target_outputs.append(pd.DataFrame(output), ignore_index=True)
        target_centers, target_labels, paths = [], [], []
        for label, group_df in target_outputs.groupby(["label"]):
            output = [o for o in group_df["feature"]]
            if self.mode == "instance":
                target_centers.extend(output)
                target_labels.extend([label for _ in group_df["label"]])
                paths.extend([p for p in group_df["path"]])
            else:
                target_centers.append(np.mean(output, axis=0).tolist())
                target_labels.append(label)
                paths.append(label)
        return target_outputs, target_centers, target_labels, paths

    def _get_source_mapping(self, i):
        label = self.source_labels_mapping[i]
        if len(label) < 4:
            label = label
        else:
            label = label.split(os.sep)[-3]
        return label

    def get_result(self, target_centers, target_labels, paths, classifier):
        count, index_sum = 0, 0
        correct_char, correct_paths = {}, {}
        top_n_chars = classifier.kneighbors(target_centers, return_distance=False)
        for top_n_char, target_label, path in zip(top_n_chars, target_labels, paths):
            predicted_chars = [self._get_source_mapping(i) for i in top_n_char]
            if target_label in set(predicted_chars):
                count += 1
                correct_index = predicted_chars.index(target_label)  # the rank of target input
                index_sum += correct_index
                if correct_index not in correct_char:
                    correct_char[correct_index] = []
                    correct_paths[correct_index] = []
                correct_char[correct_index].append(target_label)
                correct_paths[correct_index].append(path)
        return correct_char, correct_paths, count, index_sum

    def predict(self, classifier=None):
        if classifier is None:
            classifier = self.get_classifier(True)
        target_outputs, target_centers, target_labels, paths = self.get_target()
        correct_char, correct_path, count, index_sum = self.get_result(target_centers, target_labels, paths, classifier)
        if self.mode == "instance":
            accuracy = count / len(target_outputs)
        else:
            accuracy = count / len(target_outputs.groupby(["label"]))
        # sort prediction result
        correct_char = {k: v for k, v in sorted(correct_char.items(), key=lambda j: j[0])}
        correct_path = {k: v for k, v in sorted(correct_path.items(), key=lambda j: j[0])}
        chars = list()
        for c in correct_char.values():
            chars.extend(c)
        keys = ["accuracy", "index_sum", "correct", "chars"]
        values = [accuracy, index_sum, correct_char, sorted(set(chars))]
        return {"%s_%s" % (self.set_type, k): v for k, v in zip(keys, values)}, correct_path
