import torch

from .position_embedding import PositionalEmbedding


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


# @torch.no_grad()
@torch.inference_mode()
def dist_to_mask(dist: torch.Tensor, short_heads: int, long_heads: int, global_heads) -> torch.Tensor:
    B, N, _ = dist.shape
    short_mask = (dist == 1).unsqueeze(1).broadcast_to(-1, short_heads, -1, -1)
    long_mask = (dist >= 1).unsqueeze(1).broadcast_to(-1, long_heads, -1, -1)
    global_mask = (dist[:, 0:1, :] >= 1).unsqueeze(1).broadcast_to(B, global_heads, N, N)
    mask = torch.cat([short_mask, long_mask, global_mask], dim=1).reshape(-1, N, N)
    mask = torch.logical_or(mask, torch.eye(N, dtype=torch.bool, device=mask.device))  
    return ~mask


class TreeTransformer(torch.nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 short_heads: int, 
                 long_heads: int, 
                 global_heads: int, 
                 dropout: float,
                 use_pe: bool = True,
                 use_mask: bool = True,
                 ) -> None:
        super().__init__()

        self.use_pe = use_pe
        self.use_mask = use_mask

        self.short_heads = short_heads
        self.long_heads = long_heads
        self.global_heads = global_heads
        
        if self.use_pe:
            self.position_embedding = PositionalEmbedding(input_size)
        # self.input_dropout = torch.nn.Dropout(p=dropout)
        
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
        # print(nodes.shape, dist.shape)
        if self.use_pe:
            nodes = self.position_embedding(nodes)
        # embedding = self.input_dropout(nodes)
        embedding = self.dense(nodes)
        
        if self.use_mask:
            mask = dist_to_mask(dist, self.short_heads, self.long_heads, self.global_heads)
        else:
            mask = None
        hidden = self.encoder(embedding, mask=mask)[0]
        return self.bn(hidden)


# class TreeTransformerNoMask(TreeTransformer):
#     def forward(self, nodes: torch.Tensor, dist: torch.Tensor):
#         embedding = self.position_embedding(nodes)
#         embedding = self.input_dropout(embedding)
#         embedding = self.dense(nodes)
#         hidden = self.encoder(embedding)[0]
#         return self.bn(hidden)
        
# class TreeTransformerTPE(TreeTransformer):
#     def forward(self, nodes: torch.Tensor, dist: torch.Tensor):
#         embedding = self.input_dropout(nodes)
#         embedding = self.dense(embedding)
#         mask = dist_to_mask(dist, self.short_heads, self.long_heads, self.global_heads)
#         hidden = self.encoder(embedding, mask=mask)[0]
#         return self.bn(hidden)
