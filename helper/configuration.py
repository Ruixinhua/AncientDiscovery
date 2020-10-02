import torch
import os
from util import tools


class Configuration:

    def _set_device(self, device_id=0):
        # set GPU device
        self.device_id = device_id
        self.device = torch.device("cuda:%s" % device_id if torch.cuda.is_available() else "cpu")

    def __init__(self, device_id=0, core="VanillaVAE", paired_chars=None, saved_path=None):
        if paired_chars is None:
            paired_chars = ["jia", "jin"]
        self.paired_chars = paired_chars
        self._set_device(device_id=device_id)
        self.core = core
        if saved_path:
            self._set_path(saved_path)

    def _set_path(self, path):
        self.saved_path = os.path.join("checkpoint", path)
        self.log_path = os.path.join("log", path)
        tools.make_dir(self.saved_path)
        tools.make_dir(self.log_path)
        self.log_file = os.path.join(self.log_path, "log.txt")
        self.best_model_path = os.path.join(self.saved_path, "model_best.pth")

    def set_path(self, path):
        self._set_path(path)


class ModelConfiguration(Configuration):

    def __init__(self, device_id=0, core="VanillaVAE", criterion="mmd", strategy="single", learning_rate=1e-3,
                 epochs=100, early_stop=30, save_period=10, loss_type="A", model_params=None, paired_chars=None,
                 level="instance", saved_path=None):
        super().__init__(device_id, core, paired_chars, saved_path)
        self.level, self.criterion, self.strategy, self.lr = level, criterion, strategy, learning_rate
        self.epochs, self.early_stop, self.save_period, self.loss_type = epochs, early_stop, save_period, loss_type
        self.model_params = model_params if model_params else {}
        params_str = "_".join([str(p) for p in model_params.values()]) if model_params else "default"
        saved_path = saved_path if saved_path else os.path.join("_".join(self.paired_chars), core+"_"+level, params_str)
        if saved_path:
            self.set_path(saved_path)
