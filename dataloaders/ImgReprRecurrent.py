from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
import numpy as np


class ImgRepr4kRecDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        """
        This is the data loader class
        :param args: contains all the program arguments from options.py
        :param dataset: the dataset object loaded from disk
        """
        super().__init__(args, dataset)

        self.load_full_img = True if args.load_full_img == 'T' else False

    @classmethod
    def code(cls):
        return 'samples_recurrent'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=False, pin_memory=True, num_workers=self.args.dataloader_workers)
        return dataloader

    def _get_train_dataset(self):
        dataset = ImgRepr4kTrainDataset(train_data=self.train,
                                        rng=self.rng,
                                        load_full_img=self.load_full_img)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        if dataset is None:
            return None
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True, num_workers=self.args.dataloader_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'test':
            if self.test is None:
                return None
            dataset = ImgRepr4kEvalDataset(eval_data=self.test, rng=self.rng, load_full_img=self.load_full_img)
        else:
            if self.val is None:
                return None
            dataset = ImgRepr4kEvalDataset(eval_data=self.val, rng=self.rng, load_full_img=self.load_full_img)
        return dataset


class ImgRepr4kTrainDataset(data_utils.Dataset):
    def __init__(self, train_data, rng, load_full_img=False):
        self.X_train = train_data['coords']
        self.Y_train = train_data['rgb']
        self.load_full_img = load_full_img
        self.rng = rng
        self.length = 1 if self.load_full_img else len(self.Y_train)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_full_img:
            x = np.vstack((np.array([0.0, 0.0, 0.0]), self.Y_train))[:-1]
            x = np.hstack((self.X_train, x))
            y = self.Y_train
        else:
            if index == 0:
                prev_val = np.array([0.0, 0.0, 0.0])
            else:
                prev_val = self.Y_train[index-1]
            x = np.hstack((self.X_train[index], prev_val))
            y = self.Y_train[index]

        return torch.FloatTensor(x), torch.FloatTensor(y)


class ImgRepr4kEvalDataset(data_utils.Dataset):
    def __init__(self, eval_data, rng, load_full_img=False):
        self.X_eval = eval_data['coords']
        self.Y_eval = eval_data['rgb']
        self.load_full_img = load_full_img
        self.rng = rng
        self.length = 1 if self.load_full_img else len(self.Y_eval)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_full_img:
            x = np.vstack((np.array([0.0, 0.0, 0.0]), self.Y_eval))[:-1]
            x = np.hstack((self.X_eval, x))
            y = self.Y_eval
        else:
            if index == 0:
                prev_val = np.array([0.0, 0.0, 0.0])
            else:
                prev_val = self.Y_eval[index-1]
            x = np.hstack((self.X_eval[index], prev_val))
            y = self.Y_eval[index]
        return torch.FloatTensor(x), torch.FloatTensor(y)