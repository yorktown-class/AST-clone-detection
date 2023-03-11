from typing import *

import torch
import math
import numpy

TreeV = List[str]
TreeE = Tuple[List[int], List[int]]
TreeVE = Tuple[TreeV, TreeE]
TreeTensor = Tuple[torch.Tensor, torch.Tensor]


def tree_VE_prune(tree_VE: TreeVE, max_node_count = 512) -> TreeVE:
    tree_V, Tree_E = tree_VE
    n = len(tree_V)
    if n <= max_node_count:
        return tree_VE

    v_out, v_in = Tree_E
    count = [0] * len(tree_V)
    for v in v_in:
        count[v] += 1
    index = numpy.argsort(count)

    unremove_index = index[n - max_node_count:]
    vid_map = list(range(n))

    pruned_V = []
    for idx, v in enumerate(tree_V):
        if idx in unremove_index:
            pruned_V.append(v)
            vid_map[idx] = len(pruned_V) - 1
    pruned_out_in = [(out_id, in_id) for out_id, in_id in zip(v_out, v_in) if out_id in unremove_index and in_id in unremove_index]
    pruned_out = [vid_map[out_id] for out_id, _ in pruned_out_in]
    pruned_in = [vid_map[in_id] for _, in_id in pruned_out_in]
    return pruned_V, (pruned_out, pruned_in)


def tree_VE_to_tensor(tree_VE: TreeVE, word2vec_cache: Dict[str, torch.Tensor] = None) -> TreeTensor:
    def word2vec(word: str):
        if word2vec_cache and word in word2vec_cache:
            return word2vec_cache[word]
        from . import word2vec
        return word2vec.word2vec(word)
    
    tree_V, tree_E = tree_VE
    
    nodes = torch.stack([word2vec(v) for v in tree_V])
    edges = torch.tensor(tree_E, dtype=torch.long)

    n, _ = nodes.shape
    _, m = edges.shape
    mask = ~torch.eye(n, dtype=torch.bool)
    for _ in range(int(math.log(m, 2))):
        mask[edges[1, :]] &= mask[edges[0, :]]
    return nodes, mask


def merge_tree_VE(tree_VE1: TreeVE, tree_VE2: TreeVE, merge_node: str) -> TreeVE:
    tree_V1, tree_E1 = tree_VE1
    tree_V2, tree_E2 = tree_VE2

    e1_src, e1_dst = tree_E1
    e2_src, e2_dst = tree_E2
    
    e1_src = [v + 1 for v in e1_src]
    e1_dst = [v + 1 for v in e1_dst]
    d = 1 + len(tree_V1)
    e2_src = [v + d for v in e2_src]
    e2_dst = [v + d for v in e2_dst]
        
    V = [merge_node] + tree_V1 + tree_V2
    E = (e1_src + e2_src, e1_dst + e2_dst)
    return V, E


def collate_tree_tensor(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_list = [nodes for nodes, mask in batch]
    mask_list = [mask for nodes, mask in batch]
    node_batch = torch.nn.utils.rnn.pad_sequence(nodes_list)

    n = node_batch.shape[0]

    mask_base = ~torch.eye(n, dtype=torch.bool)
    mask_batch = mask_base.repeat(len(batch), 1, 1)

    for idx, mask in enumerate(mask_list):
        n, m = mask.shape
        mask_batch[idx, :n, :m] = mask

    return node_batch, mask_batch