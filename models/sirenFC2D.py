from .base import BaseModel
import torch
import torch.nn as nn
from utils import fix_random_seed_as


class SirenFC2DModel(BaseModel):
    def __init__(self, args):
        """
        This is the SIREN based FC model that accepts a 2d vector
        describing the 2d coords of the expected pixel rgb values
        :param args: system wide parameters from options.py
        """
        super().__init__(args)
        fix_random_seed_as(args.model_init_seed)

        self.layer1 = nn.Linear(2, 16, bias=True)
        self.layer2 = nn.Linear(16, 256, bias=True)
        self.layer3 = nn.Linear(256, 256, bias=True)
        self.layer4 = nn.Linear(256, 16, bias=True)
        self.layer5 = nn.Linear(16, 3, bias=True)

    @classmethod
    def code(cls):
        return 'sirenFC2D'

    def forward(self, x):
        x = torch.sin(self.layer1(x))
        x = torch.sin(self.layer3(x))
        x = torch.sin(self.layer4(x))
        x = torch.sin(self.layer2(x))
        x = torch.exp(self.layer5(x))
        return x
