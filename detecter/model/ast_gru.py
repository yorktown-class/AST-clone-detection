from typing import *

import torch
from torch_geometric.nn import MessagePassing


class DAG_GRU(MessagePassing):
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__(aggr="sum", **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRUCell(input_size, hidden_size)
        self.drop = torch.nn.Dropout(p=0.3)


    def forward(self, input: torch.Tensor, edges: torch.Tensor):
        N, _ = input.shape
        _, E = edges.shape

        assert(input.shape == (N, self.input_size))
        assert(edges.shape == (2, E))

        hidden = torch.zeros((N, self.hidden_size), device=input.device)
        available_node = torch.ones(input.shape[0], dtype=torch.bool, device=input.device)

        while torch.any(available_node).item():
            available_edge = available_node[edges[0, :]] # 起点是可用node
            in_node = torch.unique(edges[1, available_edge]) # 终点集合
            leaf_mask = available_node.clone()
            leaf_mask[in_node] = False # 入度为0的点

            use_edge_mask = leaf_mask[edges[1, :]] # 终点是当前入度为0的点

            aggregate = self.propagate(edges[:, use_edge_mask], x=self.drop(hidden))[leaf_mask]
            hidden[leaf_mask] = self.gru.forward(input[leaf_mask], aggregate)

            available_node = torch.logical_xor(available_node, leaf_mask)

        return hidden


class AST_GRU(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dense = torch.nn.Linear(self.input_size, self.hidden_size)

        self.dag_grus = torch.nn.ModuleList([
            DAG_GRU(self.hidden_size, self.hidden_size)
            for _ in range(num_layers)
        ])

    def forward(self, V: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        N, _ = V.shape
        _, M = E.shape

        assert(V.shape == (N, self.input_size))
        assert(E.shape == (2, M))

        hidden = self.dense(V)
        for gru in self.dag_grus:
            hidden = gru(hidden, E)

        return hidden
