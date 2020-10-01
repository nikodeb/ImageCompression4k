from .base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.fc_sin_layers = nn.ModuleList([
                          nn.Linear(16, 256, bias=True),
                          nn.Linear(256, 256, bias=True),
                          nn.Linear(256, 256, bias=True),
                          nn.Linear(256, 256, bias=True),
                          nn.Linear(256, 256, bias=True)])

        self.layer1 = nn.Linear(2, 16, bias=True)
        self.out = nn.Linear(256, 3, bias=True)

    @classmethod
    def code(cls):
        return 'sirenFC2D'

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        for i, layer in enumerate(self.fc_sin_layers):
            x = layer(x)
            # x = F.gelu(x)
            x = torch.sin(x)
        x = torch.exp(self.out(x))
        return x
