from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils


class ImgComp4kDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        """
        This is the data loader class
        :param args: contains all the program arguments from options.py
        :param dataset: the dataset object loaded from disk
        """
        super().__init__(args, dataset)

    @classmethod
    def code(cls):
        return 'samples'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=self.args.dataloader_workers)
        return dataloader

    def _get_train_dataset(self):
        dataset = ImgComp4kTrainDataset(train_data=self.train,
                                          rng=self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True, num_workers=0)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'test':
            if self.test is None:
                return None
            dataset = ImgComp4kEvalDataset(eval_data=self.test, rng=self.rng)
        else:
            if self.val is None:
                return None
            dataset = ImgComp4kEvalDataset(eval_data=self.val, rng=self.rng)
        return dataset


class ImgComp4kTrainDataset(data_utils.Dataset):
    def __init__(self, train_data, rng):
        self.X_train = train_data['coords']
        self.Y_train = train_data['rgb']
        self.rng = rng
        self.length = len(self.Y_train)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.X_train[index]
        y = self.Y_train[index]

        return torch.LongTensor(x), torch.LongTensor(y)


class ImgComp4kEvalDataset(data_utils.Dataset):
    def __init__(self, eval_data, rng):
        self.X_eval = eval_data['coords']
        self.Y_eval = eval_data['rgb']
        self.rng = rng
        self.length = len(self.Y_eval)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.X_eval[index]
        y = self.Y_eval[index]

        return torch.LongTensor(x), torch.LongTensor(y)