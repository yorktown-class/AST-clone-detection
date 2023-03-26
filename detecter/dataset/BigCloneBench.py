import functools
import itertools
import json
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
    def __init__(self, path: str, max_node_count: int = None) -> None:
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

            self.word_dict = create_word_dict(list(itertools.chain(*nodes_list)))

            torch.save(
                {"index_list": index_list, "tree_VE_list": tree_VE_list, "word_dict": self.word_dict}, store_path
            )

        if max_node_count:
            tree_VE_list = [tree_tools.tree_VE_prune(tree_VE, max_node_count) for tree_VE in tree_VE_list]

        self.code_map = dict(zip(index_list, tree_VE_list))

    @functools.lru_cache(maxsize=1024)
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        tree_VE = self.code_map[index]
        nodes, mask = tree_tools.tree_VE_to_tensor(tree_VE, word2vec_cache=self.word_dict)
        return nodes, mask


@functools.lru_cache(maxsize=None)
def open_code_lib(path: str, max_node_count: int) -> CodeLib:
    return CodeLib(path, max_node_count)


class DataSet(data.Dataset):
    def __init__(self, raw_data_path: str, path: str, max_node_count: int = None) -> None:
        super().__init__()
        self.code_lib = open_code_lib(raw_data_path, max_node_count)
        with open(path, "r") as f:
            raw_idx_data = f.readlines()
        self.idx_data = [tuple(map(lambda s: int(s), idx_data.split())) for idx_data in raw_idx_data]

    def __len__(self) -> int:
        return len(self.idx_data)

    def __getitem__(self, index):
        lhs, rhs, result = self.idx_data[index]
        lnodes, lmask = self.code_lib[lhs]
        rnodes, rmask = self.code_lib[rhs]
        return bool(result), (lnodes, lmask), (rnodes, rmask)


def collate_fn(batch: List[Tuple[bool, Tuple, Tuple]]):
    label_list = [label for label, ltree_VE, rtree_VE in batch]
    ltree_VE_list = [ltree_VE for label, ltree_VE, rtree_VE in batch]
    rtree_VE_list = [rtree_VE for label, ltree_VE, rtree_VE in batch]
    tree_tensor_list = list(itertools.chain(*list(zip(ltree_VE_list, rtree_VE_list))))

    label_batch = torch.tensor(label_list, dtype=torch.long)
    return label_batch, *tree_tools.collate_tree_tensor(tree_tensor_list)
