import torch

from .position_embedding import PositionalEmbedding
from . import module_tools


# @torch.no_grad()
# def parents_to_dist(parents: torch.Tensor) -> torch.Tensor:
#     N, B = parents.shape
#     dist = torch.eye(N, dtype=torch.long, device=parents.device).unsqueeze(0).expand((B, N, N)).clone()
#     return dist

#     for idx, parent in enumerate(parents.flip(0)):
#         bmask = parent != -1
#         child = N - 1 - idx

#         child_dist = dist[bmask, child, :]
#         parent_dist = dist[bmask, parent[bmask], :]
#         # child_dist = child_dist.broadcast_to(parent_dist.shape)
#         parent_dist[child_dist != 0] = child_dist[child_dist != 0] + 1
#         # parent_dist[:, :, child] = 1
#         parent_dist[:, child] = 1
#         # dist[bmask, parent[bmask], child] = 1

#     return dist


@torch.no_grad()
def dist_to_mask(dist: torch.Tensor, short_heads: int, long_heads: int, global_heads) -> torch.Tensor:
    B, N, _ = dist.shape
    short_mask = (dist == 1).unsqueeze(1).broadcast_to(-1, short_heads, -1, -1)
    long_mask = (dist >= 1).unsqueeze(1).broadcast_to(-1, long_heads, -1, -1)
    global_mask = torch.ones((1, 1, 1, 1), device=dist.device, dtype=torch.bool).broadcast_to(B, global_heads, N, N)
    # global_mask = torch.ones((B, 1, N, N), device=dist.device, dtype=torch.bool)
    mask = torch.cat([short_mask, long_mask, global_mask], dim=1).reshape(-1, N, N)  
    return ~mask


class TreeTransformer(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, short_heads: int, long_heads: int, global_heads: int, dropout: float) -> None:
        super().__init__()
        self.short_heads = short_heads
        self.long_heads = long_heads
        self.global_heads = global_heads
        
        self.position_embedding = torch.nn.Sequential(
            PositionalEmbedding(input_size),
            torch.nn.Dropout(p=dropout),
        )
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Dropout(p=dropout), 
            torch.nn.LayerNorm(hidden_size),
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=short_heads + long_heads + global_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.bn = torch.nn.BatchNorm1d(hidden_size)
    
    def forward(self, nodes: torch.Tensor, dist: torch.Tensor):
        embedding = self.dense(self.position_embedding(nodes))
        
        mask = dist_to_mask(dist, self.short_heads, self.long_heads, self.global_heads)

        hidden = self.encoder(embedding, mask=mask)[0]
        return self.bn(hidden)


class TreeTransformerNoMask(TreeTransformer):
    def forward(self, nodes: torch.Tensor, dist: torch.Tensor):
        embedding = self.dense(self.position_embedding(nodes))
        hidden = self.encoder(embedding)[0]
        return self.bn(hidden)
        

module_tools.register_module("ast_transformer", TreeTransformer(128, 128, num_layers=2, short_heads=2, long_heads=4, global_heads=2, dropout=0.1))