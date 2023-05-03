import functools
import math

import torch

"""
PE(pos, 2i) = sin(pos / 10000 ^ {2i / d})
PE(pos, 2i+1) = cos(pos / 10000 ^ {2i / d})

10000 ^ {- 2i / d} = e^log_e{10000 ^ {-2i/d}} = e^{ -2i/d * log_e{10000} }
"""


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        i_pos = torch.arange(0, channels, step=2)
        loge_10000 = torch.log(torch.tensor(10000, dtype=torch.float))
        div = torch.exp(-i_pos / channels * loge_10000)

        # self.div = div
        # self.pe_default = self.calc_pe_(1024)
        self.register_buffer("div", div, persistent=False)
        self.register_buffer("pe_default", self.calc_pe_(1024), persistent=False)

    @torch.no_grad()
    def calc_pe_(self, length: int) -> torch.Tensor:
        pe = torch.zeros((length, self.channels), dtype=torch.float, device=self.div.device)
        position = torch.arange(0, length).unsqueeze(1).to(self.div.device)
        pe[:, 0::2] = torch.sin(position * self.div.unsqueeze(0))
        pe[:, 1::2] = torch.cos(position * self.div.unsqueeze(0))
        return pe

    @torch.no_grad()
    def get_pe(self, length: int) -> torch.Tensor:
        if length <= 1024:
            return self.pe_default[:length, :].clone()
        return self.calc_pe_(length)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        F = x.shape[-1]
        assert F == self.channels
        pe = self.get_pe(N)
        if x.dim() == 3:
            pe = pe.unsqueeze(1)
        return x + pe / math.sqrt(self.channels)


class TreePositionEmbedding(PositionalEmbedding):
    def __init__(self, channels) -> None:
        super().__init__(channels)

    @torch.inference_mode()
    def forward(self, parents: torch.Tensor) -> torch.Tensor:
        N = parents.shape[0]
        tree_pe = self.get_pe(N)
        for idx, parent in enumerate(parents[1:]):
            child = idx + 1
            tree_pe[child, :] = tree_pe[child, :] + tree_pe[parent, :] / 2
        return tree_pe / (2 * math.sqrt(self.channels))
