import logging
from typing import *

import torch

from .. import logger


class Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.dense = torch.nn.Linear(in_features, in_features)
        self.sigma = torch.nn.ReLU()
        self.pool = torch.nn.Linear(in_features, out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.dense(input)
        output = self.sigma(output)
        output = self.pool(output)
        return output
