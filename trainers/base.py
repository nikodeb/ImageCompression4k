import math

from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import json
from abc import *
from pathlib import Path


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = self._create_lr_scheduler()

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers, tensorboard_writer=self.writer)
        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()

        average_meter_set = AverageMeterSet()

        #TODO: Investigate this line if issues with multi worker loaders
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()

            loss = self.calculate_loss(batch)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.6f} '.format(epoch + 1, average_meter_set['loss'].avg))

            accum_iter += batch_size

        if self._needs_to_log(accum_iter):
            # tqdm_dataloader.set_description('Logging to Tensorboard')
            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch + 1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_train_info(log_data)
            self.logger_service.log_train(log_data)

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        return accum_iter

    def validate(self, epoch, accum_iter):
        if self.val_loader is None:
            return

        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                if 'accuracy' in self.args.metrics_to_log:
                    description_metrics = ['accuracy']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch + 1,
                'accum_iter': accum_iter,
                'user_embedding': self.model.embedding.user.weight.cpu().detach().numpy()
                if self.args.dump_useritem_embeddings == 'True'
                   and self.model.embedding.user is not None
                else None,
                'item_embedding': self.model.embedding.token.weight.cpu().detach().numpy()
                if self.args.dump_useritem_embeddings == 'True'
                else None,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)

    def test(self):
        if self.test_loader is None:
            return

        print('Test best model with test set!')
        if self.args.save_models_to_disk == 'True':
            if self.args.mode == 'test':
                model_root = self.args.force_load_model_from_location
            else:
                model_root = self.export_root

            best_model = torch.load(os.path.join(model_root, 'models', 'best_acc_model.pth')).get(
                'model_state_dict')
            self.model.load_state_dict(best_model)
            self.model.eval()

            average_meter_set = AverageMeterSet()
            with torch.no_grad():
                tqdm_dataloader = tqdm(self.test_loader)
                for batch_idx, batch in enumerate(tqdm_dataloader):
                    batch = [x.to(self.device) for x in batch]

                    metrics = self.calculate_metrics(batch)

                    for k, v in metrics.items():
                        average_meter_set.update(k, v)
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                          ['Recall@%d' % k for k in self.metric_ks[:3]]
                    if 'accuracy' in self.args.metrics_to_log:
                        description_metrics = ['accuracy']
                    description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    tqdm_dataloader.set_description(description)

                average_metrics = average_meter_set.averages()
                with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                    json.dump(average_metrics, f, indent=4)
                print(average_metrics)

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _create_lr_scheduler(self):
        if self.args.lr_sched_type == 'warmup_linear':
            num_epochs = self.args.num_epochs
            num_warmup_steps = self.args.num_warmup_steps

            # Code from huggingface optimisers
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_epochs - current_step) / float(max(1, num_epochs - num_warmup_steps))
                )

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)

        elif self.args.lr_sched_type == 'warmup_cos':
            num_epochs = self.args.num_epochs
            num_warmup_steps = self.args.num_warmup_steps
            num_cycles = 0.5

            # Code from huggingface optimisers
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_epochs - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, -1)

        elif self.args.lr_sched_type == 'cos':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.num_epochs, eta_min=0.0000001)

        elif self.args.lr_sched_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.decay_step, gamma=self.args.gamma)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
            BestModelLogger(writer, model_checkpoint, metric_key='loss'),
            HparamLogger(writer, args=self.args, metric_key='loss')
        ]
        val_loggers = []
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
