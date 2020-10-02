# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 17:45
# @Function      : This is the class of base trainer
from abc import abstractmethod
import torch
import torch.optim as optim
import os
from util import tools
from helper import ModelConfiguration, AncientDataset, Prediction


class BaseTrainer:
    """
    Base class for paired dataset training
    """

    def __init__(self, config: ModelConfiguration, dataset: AncientDataset):
        self.config = config
        # setup GPU device if available, move models into configured device
        self.device = self.config.device
        self.log_file = open(self.config.log_file, "a+", encoding="utf-8")
        tools.print_log("Using %s!!!" % self.device, file=self.log_file)

        # init models
        self.model, self.optimizer = self._get_model_opt()
        self.core, self.criterion = self.config.core, self.config.criterion
        self.start_epoch, self.best_acc, self.best_index_sum = 1, 0.0, 0
        # init dataset
        self.dataset = dataset
        self.target_data, self.source_data, self.labels = dataset.target_data, dataset.source_data, dataset.labels
        self.prediction_result = []
        self.add_cons = True if self.config.loss_type == "A" else False  # add mmd loss

    def _get_model_opt(self):
        model = tools.get_model_class(self.config.core, **self.config.model_params).to(self.config.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        return model, optimizer

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to "model_best.pth"
        """
        state = {
            "best_acc": self.best_acc, "best_index_sum": self.best_index_sum, "config": self.config, "epoch": epoch,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
        }
        filename = os.path.join(self.config.saved_path, "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        tools.print_log("Saving checkpoint: {} ...".format(filename), file=self.log_file)
        if save_best:
            best_path = os.path.join(self.config.saved_path, "model_best.pth")
            torch.save(state, best_path)
            tools.print_log("Saving current best: model_best.pth ...", file=self.log_file)

    def resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        tools.print_log("Loading checkpoint: {} ...".format(resume_path), file=self.log_file)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.config = checkpoint["config"] if "config" in checkpoint else self.config
        # load accuracy of checkpoint
        self.best_acc = checkpoint["best_acc"]
        self.best_index_sum = checkpoint["best_index_sum"]

        # load architecture params from checkpoint.
        self.model, self.optimizer = self._get_model_opt()
        self.model.load_state_dict(checkpoint["model"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "source_optimizer" in checkpoint:
            self.source_optimizer.load_state_dict(checkpoint["source_optimizer"])
        else:
            self.source_optimizer = self.optimizer

        tools.print_log("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch), file=self.log_file)

    @abstractmethod
    def _train_epoch(self, **kwargs):
        """
        Training logic for an epoch

        Returns: A dictionary of log that will be output

        """
        raise NotImplementedError

    @staticmethod
    def _backward(opt, loss):
        # reset the gradients back to zero
        opt.zero_grad()
        # PyTorch accumulates gradients on subsequent backward passes
        loss.backward()
        opt.step()

    def predict(self):
        pred = Prediction(self.dataset.target_val, self.dataset.labels_val, self.dataset.source_data_full,
                          self.dataset.source_labels_full, self.dataset.paths_val, core=self.core,
                          mode=self.config.level)
        pred.set_model(self.model)
        val_result, val_paths = pred.predict(pred.get_classifier(False))
        self.prediction_result.append([val_result, val_paths])
        return val_result

    def train(self, resume_path=None):
        """
        Full training logic
        """
        if resume_path is not None:
            self.resume_checkpoint(resume_path)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            log = {"epoch": "%s/%s" % (epoch, self.config.epochs)}
            self.model.train()
            # save logged information into log dict
            log.update(self._train_epoch())
            best = False
            self.model.eval()
            val_result = self.predict()
            log.update(val_result)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            cur_acc, cur_index_sum = val_result["val_accuracy"], val_result["val_index_sum"]
            # check whether models performance improved or not
            improved = (cur_acc > self.best_acc) or \
                       (cur_acc == self.best_acc and cur_index_sum < self.best_index_sum)

            if improved:
                not_improved_count, self.best_acc, self.best_index_sum, best = 0, cur_acc, cur_index_sum, True
                self._save_checkpoint(epoch, save_best=best)
            else:
                not_improved_count += 1

            # print logged information to the screen
            for key, value in log.items():
                tools.print_log("{:30s}: {}".format(str(key), value), file=self.log_file)

            if not_improved_count > self.config.early_stop:
                tools.print_log("Validation performance did not improve for %s epochs.So Stop" % self.config.early_stop,
                                file=self.log_file)
                break

            if epoch % self.config.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
