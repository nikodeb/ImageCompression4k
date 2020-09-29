from .ImgComp4k import ImgComp4KDataset

DATASETS = {
    ImgComp4KDataset.code(): ImgComp4KDataset
}


def dataset_factory(args):
    """
    Load the specified dataset object
    :param args: system wide arguments from options.py
    :return: dataset object
    """
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
