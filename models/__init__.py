from .sirenFC2D import SirenFC2DModel

MODELS = {
    SirenFC2DModel.code(): SirenFC2DModel,
}


def model_factory(args):
    """
    Load the specified model
    :param args: system wide arguments from options.py
    :return: architecture
    """
    model = MODELS[args.model_code]
    return model(args)
