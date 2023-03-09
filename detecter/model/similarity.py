from typing import *

import torch
import logging

from .ast_attention import AstAttention

logger = logging.getLogger("similarity")

# class Similarity(torch.nn.Module):
#     def __init__(self, in_features, num_heads) -> None:
#         super().__init__()
#         self.in_features = in_features
#         self.query = torch.nn.Parameter(torch.Tensor(1, in_features))
#         self.attn = torch.nn.MultiheadAttention(in_features, num_heads=num_heads)
#         self.norm = torch.nn.LayerNorm(in_features)

#         self.dense = torch.nn.Linear(in_features * 2, in_features * 2)
#         self.pool = torch.nn.Linear(in_features * 2, 2)

#     def forward(self, l_tree: torch.Tensor, r_tree: torch.Tensor) -> torch.Tensor:
#         print(l_tree.shape)
#         print(r_tree.shape)
#         L1, B, F = l_tree.shape
#         L2, _, _ = r_tree.shape
#         assert(r_tree.shape == (L2, B, F))

#         query = self.query.repeat(1, B, 1)
#         l_feature, _ = self.attn(query, l_tree, l_tree)
#         l_feature.reshape(B, F)
#         l_feature = self.norm(l_feature)
#         r_feature, _ = self.attn(query, r_tree, r_tree)
#         r_feature.reshape(B, F)
#         r_feature = self.norm(r_feature)

#         feature = torch.cat([l_feature, r_feature], dim=1)
#         feature = self.dense(feature)
#         feature = torch.relu(feature)
#         output = self.pool(feature)

#         return output

class Similarity(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.encoder = AstAttention(input_size, hidden_size, num_heads, num_heads)
        self.query = torch.nn.Parameter(torch.Tensor(input_size))
    
    def forward(self, input1: torch.Tensor, mask1: torch.Tensor, input2: torch.Tensor, mask2: torch.Tensor):
        print(input1.shape)
        print(mask1.shape)
        print(input2.shape)
        print(mask2.shape)
        L1, B, F = input1.shape
        L2, B, _ = input2.shape
        assert(mask1.shape == (B, L1, L1))
        assert(input2.shape == (L2, B, F))
        assert(mask2.shape == (B, L2, L2))

        query = self.query.repeat(1, B, 1)
        input = torch.cat([input1, input2, query], dim=0)
        mask = torch.ones((B, L1 + L2 + 1, L1 + L2 + 1), device=input.device)

        mask[:, 0: L1, 0: L1] = mask1
        mask[:, L1: L1 + L2, L1: L1 + L2] = mask2
        mask[:, -1, :] = False

        return self.encoder(input, mask)
