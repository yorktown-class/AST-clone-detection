from typing import *

import torch
from torch.utils import data
import pandas
import itertools

from .import tree_tools


Batch = Tuple[bool, tree_tools.TreeTensor, tree_tools.TreeTensor]

class PairCodeset(data.Dataset):
    def __init__(self, code_data: pandas.DataFrame, pair: pandas.DataFrame) -> None:
        super().__init__()
        self.code_data = code_data
        self.pair = pair
        self.max_node_count = None
        self.tpe = False
    
    def __len__(self) -> int:
        return len(self.pair)

    def drop(self, max_node_count: int):
        if max_node_count is None:
            return
        def check_length(series):
            return (
                tree_tools.peep_tree_nodes(self.code_data.loc[series["lhs"], "tree"]) <= max_node_count and
                tree_tools.peep_tree_nodes(self.code_data.loc[series["rhs"], "tree"]) <= max_node_count                 
            )
        self.pair = self.pair[self.pair.apply(check_length, axis=1)]
        return self
    
    def prune(self, max_node_count: int):
        self.max_node_count = max_node_count
        return self
    
    def sample(self, n: int):
        self.pair = self.pair.sample(n)
        return self
    
    def use_tpe(self, enable: bool = True):
        self.tpe = enable
        return self
    
    @torch.inference_mode()
    def gettree_(self, code_idx) -> tree_tools.TreeTensor:
        tree = self.code_data.loc[code_idx, "tree"]
        tree_tensor = tree_tools.tree_to_tensor(tree, add_tree_position_embedding=self.tpe)
        if self.max_node_count:
            tree_tensor = tree_tools.prune_tree_tensor(tree_tensor, self.max_node_count)
        return tree_tensor

    def __getitem__(self, idx) -> Batch:
        lhs, rhs, label = self.pair.iloc[idx]
        return label, self.gettree_(lhs), self.gettree_(rhs)


@torch.inference_mode()
def collate_fn(batch: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_batch =  [label for label, _, _ in batch]
    tree_batch = list(itertools.chain(*[(ltree, rtree) for _, ltree, rtree in batch]))
    return torch.tensor(label_batch, dtype=torch.bool), *tree_tools.collate_tree_tensor(tree_batch)
