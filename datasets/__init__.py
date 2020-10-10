from .ImgRepr import ImgRepr4KDataset

DATASETS = {
    ImgRepr4KDataset.code(): ImgRepr4KDataset
}


def dataset_factory(args):
    """
    Load the specified dataset object
    :param args: system wide arguments from options.py
    :return: dataset object
    """
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
