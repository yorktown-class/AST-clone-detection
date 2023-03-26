import functools
import itertools
import json
import random
from typing import *

import torch
from torch.utils import data

from .. import parser, tree_tools
from ..word2vec import create_word_dict


@functools.lru_cache(maxsize=None)
def read_raw_data(path: str) -> Dict:
    """
    从data.jsonl.txt中读取数据, 并解析为Dict。
    其中key是一条数据的"idx", value是此条数据的"func"
    """
    result = dict()
    with open(path, "r") as f:
        data_list = f.readlines()
    for raw_data in data_list:
        data = json.loads(raw_data)
        index = int(data["idx"])
        func = data["func"]
        result[index] = func
    return result


class CodeLib:
    def __init__(self, path: str) -> None:
        store_path = path + ".pt"

        try:
            data = torch.load(store_path)
            index_list = data["index_list"]
            tree_VE_list = data["tree_VE_list"]
            self.word_dict = data["word_dict"]

        except IOError:
            index_list = []
            code_list = []

            with open(path, "r") as f:
                data_list = f.readlines()

            for raw_data in data_list:
                data = json.loads(raw_data)
                index = int(data["idx"])
                func = data["func"]
                index_list.append(index)
                code_list.append(func)

            tree_VE_list = list(map(lambda s: parser.parse(s, "java"), (code for code in code_list)))
            nodes_list = [tree_V for tree_V, tree_E in tree_VE_list]

            self.word_dict = create_word_dict(list(itertools.chain(*nodes_list)) + ["<CODE_COMPARE>"])

            torch.save(
                {"index_list": index_list, "tree_VE_list": tree_VE_list, "word_dict": self.word_dict}, store_path
            )

        self.code_map = dict(zip(index_list, tree_VE_list))

    def __getitem__(self, index) -> tree_tools.TreeVE:
        tree_VE = self.code_map[index]
        return tree_VE


@functools.lru_cache(maxsize=None)
def open_code_lib(path: str) -> CodeLib:
    return CodeLib(path)


class DataSet(data.Dataset):
    def __init__(self, raw_data_path: str, path: str, max_node_count: int = None, fixed_prune: bool = False) -> None:
        super().__init__()
        self.max_node_count = max_node_count
        self.code_lib = open_code_lib(raw_data_path)
        self.seed = hash(raw_data_path) if fixed_prune else None
        with open(path, "r") as f:
            raw_idx_data = f.readlines()
        self.idx_data = [tuple(map(lambda s: int(s), idx_data.split())) for idx_data in raw_idx_data]

    def __len__(self) -> int:
        return len(self.idx_data)

    def __getitem__(self, index):
        lhs, rhs, result = self.idx_data[index]
        tree_VE = tree_tools.merge_tree_VE(self.code_lib[lhs], self.code_lib[rhs], "<CODE_COMPARE>")
        tree_VE = tree_tools.tree_VE_prune(tree_VE, self.max_node_count, self.seed)
        return bool(result), *tree_tools.tree_VE_to_tensor(tree_VE, self.code_lib.word_dict)


def collate_fn(batch: List[Tuple[bool, torch.Tensor, torch.Tensor]]):
    label_list = [label for label, nodes, mask in batch]
    tree_tensor_list = [(nodes, mask) for label, nodes, mask in batch]
    return torch.tensor(label_list, dtype=torch.long), *tree_tools.collate_tree_tensor(tree_tensor_list)
