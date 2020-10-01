from .base import AbstractTrainer

import torch.nn as nn


class ImgComp4kTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

        self.mse = nn.MSELoss(reduction='mean')

    @classmethod
    def code(cls):
        return 'samples'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        #TODO: loss might need to be calculated on cpu instead of gpu - benchmark it
        coords, rgb_labels = batch
        rgb_preds = self.model(coords)
        final_loss = self.mse(rgb_preds, rgb_labels)
        return final_loss

    def calculate_metrics(self, batch):
        #TODO: loss might need to be calculated on cpu instead of gpu - benchmark it
        coords, rgb_labels = batch
        rgb_preds = self.model(coords)
        final_loss = self.mse(rgb_preds, rgb_labels)
        metrics = {'loss': final_loss}
        return metrics
