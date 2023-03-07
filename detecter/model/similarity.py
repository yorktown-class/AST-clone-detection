from typing import *

import torch


class Similarity(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.dense = torch.nn.Linear(in_features * 2, in_features * 2)
        self.sigma = torch.nn.ReLU()
        self.pool = torch.nn.Linear(in_features * 2, 1)

    def forward(self, l_feature: torch.Tensor, r_feature: torch.Tensor) -> torch.Tensor:
        assert(l_feature.shape == r_feature.shape)
        assert(l_feature.shape[-1] == self.in_features)

        # feature = l_feature * r_feature
        # output: torch.Tensor = self.dense(feature)
        feature = torch.cat([l_feature, r_feature], dim=1)
        feature = self.dense(feature)
        feature = self.sigma(feature)
        output = self.pool(feature)

        return output.reshape(-1)
