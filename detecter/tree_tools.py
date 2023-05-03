from typing import *

import numpy
import torch

from . import pace
from .position_embedding import TreePositionEmbedding

Nodes = numpy.ndarray
Parents = numpy.ndarray
Tree = Tuple[Nodes, Parents]
TreeTensor = Tuple[torch.Tensor, torch.ShortTensor]


@torch.inference_mode()
def nodes_to_tensor(nodes: Nodes) -> torch.Tensor:
    node_tensor = torch.zeros((len(nodes), 128))
    for idx, word in enumerate(nodes):
        x = pace.position_aware_char_embedding(word)
        node_tensor[idx, :] = x
    return node_tensor


@torch.inference_mode()
def parents_to_dist(parents: Parents) -> torch.Tensor:
    n = len(parents)
    dist = torch.eye(n, dtype=torch.short)
    for idx, parent in enumerate(parents[:0:-1]):
        child = n - idx - 1
        mask = dist[child, :] != 0
        dist[parent, mask] = dist[child, mask] + 1
        dist[parent, child] = 1
    assert dist[0][0] == 1
    return dist


tree_position_embedding = TreePositionEmbedding(128)


@torch.inference_mode()
def parents_to_tpe(parents: Parents) -> torch.Tensor:
    return tree_position_embedding(torch.tensor(parents))


@torch.inference_mode()
def tree_to_tensor(tree: Tree, add_tree_position_embedding: bool = False) -> TreeTensor:
    nodes, parents = tree
    nodes_tensor = nodes_to_tensor(nodes)
    dist_tensor = parents_to_dist(parents)
    if add_tree_position_embedding:
        nodes_tensor += parents_to_tpe(parents)
    return (nodes_tensor, dist_tensor)


def peep_tree_nodes(tree: Tree) -> int:
    nodes, _ = tree
    return len(nodes)


# def prune_tree_tensor(tree_tensor: TreeTensor, max_node_count: int) -> TreeTensor:
#     nodes, parents = tree_tensor
#     n = nodes.shape[0]

#     if n <= max_node_count:
#         return tree_tensor

#     remove_count = n - max_node_count
#     removed = torch.multinomial(torch.arange(1, n, 1), remove_count)
#     unremoved_mask = torch.ones(n, dtype=torch.bool)
#     unremoved_mask[removed] = False

#     node_id_map = torch.cumsum(unremoved_mask, dtype=torch.long) - 1

#     nodes = nodes[unremoved_mask]
#     update_parents = torch.arange(0, n, 1)
#     update_parents[~unremoved_mask] = update_parents[update_parents[~unremoved_mask]]

#     parents = parents[unremoved_mask]
#     parents[1:] = node_id_map[parents[1:]]

#     return (nodes, parents)


@torch.inference_mode()
def prune_tree_tensor(tree_tensor: TreeTensor, max_node_count: int) -> TreeTensor:
    nodes, dist = tree_tensor
    n = nodes.shape[0]

    if n <= max_node_count:
        return tree_tensor

    k = n - max_node_count
    removed = torch.multinomial(torch.ones(n - 1), k) + 1
    maintain_mask = torch.ones(n, dtype=torch.bool)
    maintain_mask[removed] = False

    nodes = nodes[maintain_mask]
    dist = dist[maintain_mask, :]
    dist = dist[:, maintain_mask]

    return (nodes, dist)


@torch.inference_mode()
def collate_tree_tensor(batch: List[TreeTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_list = [nodes for nodes, dist in batch]
    dist_list = [dist for nodes, dist in batch]
    node_batch = torch.nn.utils.rnn.pad_sequence(nodes_list)
    N, B, _ = node_batch.shape
    dist_batch = torch.eye(N, dtype=torch.short).unsqueeze(0).expand((B, N, N)).clone()
    for idx, dist in enumerate(dist_list):
        dist_batch[idx, : dist.shape[0], : dist.shape[1]] = dist
    return node_batch, dist_batch
