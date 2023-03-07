from typing import *

import torch
from torch.nn import Embedding, Linear
from torch_geometric.nn import (
    HeteroConv, GCNConv, SAGEConv, GATConv, Linear, 
     PositionalEncoding, GatedGraphConv)
     
from torch_geometric.data import HeteroData



class AST_LSTM(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.EMBEDDING_SIZE = 384

        # self.var_embedding = Embedding(1**64, 128, sparse=True)
        self.var_embedding = PositionalEncoding(128)
        self.word_bag = dict() # hash -> int
        self.code_embedding = Embedding(200, 384)
        self.linear = Linear(384, 384)

        self.conv = GatedGraphConv(self.EMBEDDING_SIZE, 1, aggr='mean')

    def forward(self, V: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        N = V.shape[0]
        M = E.shape[1]
        
        hidden = V.clone()
        edges = E.clone()

        iterator_count = 0
        while edges.shape[1] and iterator_count < 15:
            iterator_count += 1

            nleaf = torch.unique(edges[1, :])
            leaf_mask = torch.ones(N, dtype=torch.bool)
            leaf_mask[nleaf] = False # 叶节点以及孤立点，但孤立点不影响后续

            use_edge_mask = leaf_mask[edges[0, :]]
            use_node = torch.unique(edges[:, use_edge_mask])
            
            use_node_mask = torch.zeros(N, dtype=torch.long)
            use_node_mask[use_node] = 1
            index_map = torch.cumsum(use_node_mask, dim=0).to(device=edges.device) - 1

            # [n, EMB]
            # use_node_index = torch.Tensor(use_node.shape[0], self.EMBEDDING_SIZE)
            # use_node_index[:, :] = use_node
            
            # hidden[use_node] = self.conv.forward(hidden[use_node], index_map[edges[:, use_edge_mask]])
            hidden = self.conv.forward(hidden, index_map[edges[:, use_edge_mask]])
            
            # hidden = hidden.scatter(0, use_node_index, 
            #                   self.conv.forward(hidden[use_node], index_map[edges[:, use_edge_mask]]))
            # print(hidden.grad_fn)
            edges = edges[:, ~use_edge_mask]
        
        
        print("h", hidden[0])
        return hidden


