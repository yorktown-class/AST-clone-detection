import random
from typing import *

import torch

TreeV = List[str]
TreeE = Tuple[List[int], List[int]]
TreeVE = Tuple[TreeV, TreeE]
TreeTensor = Tuple[torch.Tensor, torch.Tensor]


def tree_VE_hash(tree_VE: TreeVE) -> int:
    tree_V, tree_E = tree_VE
    return hash((tuple(tree_V), tuple(tree_E[0]), tuple(tree_E[1])))


def tree_VE_prune(tree_VE: TreeVE, max_node_count: int = 512, seed: int = None) -> TreeVE:
    """
    递归调用,随机删去叶子,直到树的节点数小于max_node_count
    """
    tree_V, Tree_E = tree_VE
    n = len(tree_V)
    if n <= max_node_count:
        return tree_VE

    v_out, v_in = Tree_E
    count = [0] * len(tree_V)
    for v in v_in:
        count[v] += 1

    can_prune = [i for i in range(n) if count[i] == 0]
    rand = random.Random(seed) if seed else random
    rand.shuffle(can_prune)

    k = min(len(can_prune), n - max_node_count)
    pruned = set(can_prune[:k])

    pruned_V = []
    vid_map = list(range(n))
    for idx, v in enumerate(tree_V):
        if idx not in pruned:
            pruned_V.append(v)
            vid_map[idx] = len(pruned_V) - 1

    pruned_out_in = [(out_id, in_id) for out_id, in_id in zip(v_out, v_in) if out_id not in pruned]
    pruned_out = [vid_map[out_id] for out_id, _ in pruned_out_in]
    pruned_in = [vid_map[in_id] for _, in_id in pruned_out_in]

    assert len(pruned_V) - 1 == len(pruned_in)
    assert len(pruned_V) < len(tree_V)

    return tree_VE_prune((pruned_V, (pruned_out, pruned_in)), max_node_count)


def tree_VE_to_tensor(tree_VE: TreeVE, word2vec_cache: Dict[str, torch.Tensor] = None) -> TreeTensor:
    """
    将由(V, E)表示的树转换为由(nodes, mask)两个tensor表示的树。

    其中nodes表示点矩阵, mask为可达矩阵取反
    """

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
    mask = torch.eye(n, dtype=torch.bool)

    for i in range(m):
        mask[edges[1, i]] = torch.logical_or(mask[edges[1, i]], mask[edges[0, i]])

    mask = ~mask
    return nodes, mask


def merge_tree_VE(tree_VE1: TreeVE, tree_VE2: TreeVE, merge_node: str) -> TreeVE:
    """
    通过merge_node合并两颗树
    """
    tree_V1, tree_E1 = tree_VE1
    tree_V2, tree_E2 = tree_VE2

    e1_src, e1_dst = tree_E1
    e2_src, e2_dst = tree_E2

    e1_src = [v + 1 for v in e1_src] + [1]
    e1_dst = [v + 1 for v in e1_dst] + [0]
    d = 1 + len(tree_V1)
    e2_src = [v + d for v in e2_src] + [d]
    e2_dst = [v + d for v in e2_dst] + [0]

    V = [merge_node] + tree_V1 + tree_V2
    E = (e1_src + e2_src, e1_dst + e2_dst)
    assert len(V) - 1 == len(E[0])
    return V, E


def collate_tree_tensor(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_list = [nodes for nodes, mask in batch]
    mask_list = [mask for nodes, mask in batch]
    node_batch = torch.nn.utils.rnn.pad_sequence(nodes_list)

    n = node_batch.shape[0]

    # mask_base = ~torch.eye(n, dtype=torch.bool)
    # mask_batch = mask_base.repeat(len(batch), 1, 1)
    mask_batch = torch.ones(len(mask_list), n, n)

    for idx, mask in enumerate(mask_list):
        mask_batch[idx, : mask.shape[0], : mask.shape[1]] = mask

    return node_batch, mask_batch
