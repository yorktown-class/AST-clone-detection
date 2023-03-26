import logging
from typing import *

import torch

from .ast_attention import AstAttention

logger = logging.getLogger("similarity")


class Similarity(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.in_features = in_features

        self.norm = torch.nn.LayerNorm(in_features)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, l_feature: torch.Tensor, r_feature: torch.Tensor) -> torch.Tensor:
        l_feature = self.norm(l_feature)
        r_feature = self.norm(r_feature)
        mul = torch.sum(l_feature * r_feature, dim=-1)
        return mul + self.bias
