from datasets import dataset_factory
from .ImgComp4k import ImgComp4kDataloader


DATALOADERS = {
    ImgComp4kDataloader.code(): ImgComp4kDataloader
}


def dataloader_factory(args):
    """
    This method loads the specified dataset using the dataset factory and returns the three data loaders
    :param args: system wide arguments from options.py
    :return: train, validation, test data loaders
    """
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
