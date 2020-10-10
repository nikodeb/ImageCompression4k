from .sirenFC2D import SirenFC2DModel
from .siren import Siren

MODELS = {
    SirenFC2DModel.code(): SirenFC2DModel,
    Siren.code(): Siren,
}


def model_factory(args):
    """
    Load the specified model
    :param args: system wide arguments from options.py
    :return: architecture
    """
    model = MODELS[args.model_code]
    return model(args)
